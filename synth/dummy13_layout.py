"""Dummy13（CC BY 4.0 の印刷用 STL）を T ポーズに組み立てる配置計算。

各 STL は CAD の関節中心（ボールの中心・ソケットの中心・ヒンジピンの軸）が原点付近に来るように
書き出されているので、パーツ同士の「どの点とどの点が一致するか」から配置を計算できる。
下の寸法は assets/dummy13/*.stl を実測した値（mm）。

座標系（ワールド）: Z 上, フィギュアの正面は -Y, フィギュアの左手側が +X（Blender の慣習どおり）。
単位は mm。Blender に読み込むときに 0.001 倍してメートルにする。

numpy だけに依存するので Blender 内でも通常の Python でも使える。
"""

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------- 実測した特徴点（パーツのローカル座標, mm）

BALL_R = 3.0  # ボールジョイントの半径（全パーツ共通）

F = {
    # 胴体（ローカル Y が上、Z が厚み方向）
    "hips.center_ball": (0, 0, 0), "hips.ball_L": (8, 0, 0), "hips.ball_R": (-8, 0, 0),
    "waist.socket": (0, 0, 0), "waist.ball": (0, 11, 0),
    "abdomen.socket": (0, 0, 0), "abdomen.ball": (0, 13, 0),
    "chest.socket": (0, 0, 0), "chest.ball_L": (6, 14, 0), "chest.ball_R": (-6, 14, 0), "chest.neck_socket": (0, 20, 0),
    "neck.ball_low": (0, 0, 0), "neck.ball_mid": (0, 5, 0), "neck.ball_top": (0, 10, 0),
    "head.socket": (0, 0, 2.5),
    "clavicle.socket": (0, 0, 0), "clavicle.ball": (0, 7, 0),
    # 手足（ローカル Y が関節の並ぶ方向、Z がヒンジ軸、厚み中心は z=2.5）
    "hs.socket": (0, 0, 2.5),               # frame, hip and shoulder（股関節・肩のソケット）
    "thigh.pin": (0, 0, 2.5), "thigh.hs": (0, 27.2, 0),      # hs の原点が来る位置（上端のスロット）
    "upper_arm.pin": (0, 0, 2.5), "upper_arm.hs": (0, 19.2, 0),
    "hinge.clip1": (0, 0, 1.4), "hinge.clip2": (6, 0, 1.4),  # frame, knee and elbow（二重ヒンジ）
    "hinge.armor_pin": (3, 3.524, 1.4),
    "shin.pin": (0, 0, 2.5), "shin.ball": (0, -28, 2.5),
    "forearm.pin": (0, 0, 2.5), "forearm.ball": (0, -15, 2.5),
    "ankle.socket": (0, 0, 2.5), "ankle.pin": (0, -5.428, 2.5),
    "hand.socket": (0, 0, 0),
}

STL = {
    "hips": "frame, hips", "waist": "frame, waist", "abdomen": "frame, abdomen", "chest": "frame, chest",
    "neck": "frame, neck", "head": "frame, head", "clavicle": "frame, clavicle, 2x",
    "hs": "frame, hip and shoulder, 4x", "thigh": "frame, thigh, 2x", "upper_arm": "frame, upper arm, 2x",
    "hinge": "frame, knee and elbow, 4x", "shin": "frame, shin, 2x", "forearm": "frame, forearm, 2x",
    "ankle": "frame, ankle, 2x",
    "hand": "hand, {pose}, left",  # 右は "right"。pose は open / fist / grip
}


# ---------------------------------------------------------------- 行列ヘルパ


def basis(x=None, y=None, z=None) -> np.ndarray:
    """ローカル軸のうち2本のワールド方向を指定して回転行列を作る（右手系）。"""
    x, y, z = (None if v is None else np.asarray(v, float) for v in (x, y, z))
    if x is None:
        x = np.cross(y, z)
    elif y is None:
        y = np.cross(z, x)
    elif z is None:
        z = np.cross(x, y)
    return np.column_stack([x, y, z])


def mat(R: np.ndarray, t) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, t
    return M


def place(R: np.ndarray, local_pt, world_pt) -> np.ndarray:
    """回転 R のパーツを、ローカル点 local_pt がワールド点 world_pt に来るように置く。"""
    return mat(R, np.asarray(world_pt, float) - R @ np.asarray(local_pt, float))


def apply(M: np.ndarray, p) -> np.ndarray:
    return M[:3, :3] @ np.asarray(p, float) + M[:3, 3]


WORLD_MIRROR = np.diag([-1.0, 1.0, 1.0, 1.0])


def reflect(axis: int, at: float = 0.0) -> np.ndarray:
    """ローカル座標で axis=at の平面について鏡映する 4x4 行列。"""
    S = np.eye(4)
    S[axis, axis] = -1.0
    S[axis, 3] = 2 * at
    return S


# 右半身は「左半身をワールドで鏡映」＋「パーツ自身の対称面で鏡映」で作る（2回の鏡映で正しい回転になる）。
# 左右で同じ STL を使うパーツは、どこかの平面について対称なのでこれで実物どおりの形になる。
SYMMETRY = {
    "clavicle": reflect(0), "hs": reflect(0), "thigh": reflect(0), "upper_arm": reflect(0),
    "shin": reflect(0), "forearm": reflect(0),
    "ankle": reflect(2, 2.5),  # 足首は厚み方向で対称（左右で鏡写しになる向き。実物で確認）
    "hinge": reflect(2, 1.4),  # 「3」の形は厚み方向の中央面で対称
    "hand": reflect(0),         # 右手は左手を鏡映した別ファイル
}


# ---------------------------------------------------------------- 組み立て


@dataclass
class Part:
    name: str          # インスタンス名（例 "thigh.L"）
    stl: str           # assets/dummy13/ のファイル名（拡張子なし）
    matrix: np.ndarray  # ワールド行列（mm）
    bone: str          # 親子付けするボーン
    sym: np.ndarray = None  # 右半身を作るときのパーツ自身の対称面（鏡映行列）


# アーマーの取り付け位置: STL → (取り付け先パーツ, 回転, 平行移動)。取り付け先パーツのローカル座標（mm）。
# 自動探索（アーマーがフレームを包み、めり込まない位置）で求め、向きは組み立てガイドと照合して選んだ。
ARMOR = {
    "armor, head": ("head", [[0, 0, -1], [1, 0, 0], [0, -1, 0]], [5.75, 1.13, 3.35]),
    "armor, outer chest": ("chest", [[-1, 0, 0], [0, 0, -1], [0, -1, 0]], [0.0, 20.0, 0.0]),  # 「13」が正面、下の縁にアーチ。
    # 高さは内側の球状のくぼみ（底から 5.95）を肩のボールの中心（chest y=14）に合わせた
    "armor, inner chest": ("chest", [[1, 0, 0], [0, 0, -1], [0, 1, 0]], [0.0, 10.0, 0.0]),  # 外側の中に隠れ、肩のソケットの下（実物で確認）
    "armor, abdomen": ("abdomen", [[1, 0, 0], [0, 0, 1], [0, -1, 0]], [0.0, 2.0, 0.0]),  # 丸い面が上、背面のパネル下端に面取り。高さは見た目で調整
    "armor, waist": ("waist", [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], [-6.6, -3.12, 10.0]),  # 後ろにスタンド用の穴（ガイド手順26）
    "armor, crotch": ("waist", [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [-0.0, 0.03, -0.0]),
    "armor, hip, 2x": ("hip_joint.L", [[0, -1, 0], [0, 0, 1], [-1, 0, 0]], [6.0, -5.25, 2.55]),
    "armor, thigh, 2x": ("thigh.L", [[1, 0, 0], [0, 0, -1], [0, 1, 0]], [-6.0, 22.0, -1.9]),
    "armor, shin, 2x": ("shin.L", [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], [-3.54, -23.1, 6.05]),  # 前後は実物で確認
    "armor, foot, 2x": ("ankle.L", [[0, -1, 0], [0, 0, 1], [-1, 0, 0]], [0.0, -8.9, 8.1]),
    "armor, shoulder, 2x": ("shoulder_joint.L", [[0, 1, 0], [0, 0, 1], [1, 0, 0]], [-4.45, -5.03, -0.75]),
    "armor, upper arm, 2x": ("upper_arm.L", [[0, 0, -1], [0, 1, 0], [1, 0, 0]], [3.3, 1.2, -1.55]),  # 肩のアーマーのすぐ下（実物写真で確認）
    "armor, forearm, 2x": ("forearm.L", [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], [-3.35, -13.6, 6.0]),  # 斜めの出っ張りの面が上下（肘のヒンジ軸と垂直）、下側の段が肘の外側（実物写真で確認）
    # 膝のカバー: 裏の溝（半径1.52）を膝パーツ上部の輪（半径1.50）にはめる
    "armor, knee, 2x": ("knee.L", [[0, 1, 0], [0, 0, 1], [1, 0, 0]], None),
}
# 首のアーマー: 内側の球状の受け（半径 2.5）を首パーツ中央の球にはめ、後ろから差し込む向き
# （C の字は水平面内、開口が前、段差のある面が下。実物で確認済み）。中央の四角い突起が切り欠きに入る。
NECK_ARMOR = ([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], (3.5, -0.9, -1.1))  # (首パーツ基準の回転, アーマー側の球の中心)
KNEE_ARMOR_CLIP = ((2.5, 5.0, 0.99), (3.0, 3.55, 1.4))  # (アーマー側の溝の中心, 膝パーツ側の輪の中心)
# つま先: 丸い端（関節）を足のアーマー前端の丸い部分に合わせる（足アーマーのローカル座標）
TOE = ([[0, -1, 0], [1, 0, 0], [0, 0, 1]], (0.0, -1.5, 2.0), (14.5, 0.0, 2.0))
TOE_TIP = (0.0, -7.0, 0.5)
# 左右で同じ STL を使うアーマーの対称面
ARMOR_SYMMETRY = {
    "armor, hip, 2x": reflect(1, 6.0), "armor, thigh, 2x": reflect(1, 4.5), "armor, knee, 2x": reflect(0, 2.5),
    "armor, shin, 2x": reflect(0, 3.5), "armor, foot, 2x": reflect(1, 0.0), "armor, toe, 2x": reflect(0, 0.0),
    "armor, shoulder, 2x": reflect(1, 4.5), "armor, upper arm, 2x": reflect(0, 4.0), "armor, forearm, 2x": reflect(0, 3.5),
}

# ボーン: 名前 → (根元の関節, 先端の関節, 親)。関節名は build() の joints。
# 膝・肘は二重ヒンジなので、ヒンジ部品（knee / elbow）にも短いボーンを割り当てる。
BONES = {
    "hips": ("hips", "hips_tail", None),
    "waist": ("hips", "waist_top", "hips"),
    "abdomen": ("waist_top", "abdomen_top", "waist"),
    "chest": ("abdomen_top", "neck_base", "abdomen"),
    "neck": ("neck_base", "neck_top", "chest"),
    "head": ("neck_top", "head_top", "neck"),
}
for _s in ("L", "R"):
    BONES.update({
        f"clavicle.{_s}": (f"clavicle_base.{_s}", f"shoulder.{_s}", "chest"),
        f"upper_arm.{_s}": (f"shoulder.{_s}", f"elbow1.{_s}", f"clavicle.{_s}"),
        f"elbow.{_s}": (f"elbow1.{_s}", f"elbow2.{_s}", f"upper_arm.{_s}"),
        f"forearm.{_s}": (f"elbow2.{_s}", f"wrist.{_s}", f"elbow.{_s}"),
        f"hand.{_s}": (f"wrist.{_s}", f"hand_tip.{_s}", f"forearm.{_s}"),
        f"thigh.{_s}": (f"hip.{_s}", f"knee1.{_s}", "hips"),
        f"knee.{_s}": (f"knee1.{_s}", f"knee2.{_s}", f"thigh.{_s}"),
        f"shin.{_s}": (f"knee2.{_s}", f"ankle.{_s}", f"knee.{_s}"),
        f"ankle.{_s}": (f"ankle.{_s}", f"foot_pin.{_s}", f"shin.{_s}"),
        f"foot.{_s}": (f"foot_pin.{_s}", f"toe.{_s}", f"ankle.{_s}"),
    })


@dataclass
class Layout:
    parts: list = field(default_factory=list)
    joints: dict = field(default_factory=dict)  # 名前 → ワールド座標（mm）

    def add(self, name, stl, M, bone, sym=None):
        self.parts.append(Part(name, stl, M, bone, sym))
        return M


def add_armor(L: "Layout", side: str) -> None:
    """ARMOR の定義どおりにアーマーを置く（胴体は side に関係なく1回だけ）。"""
    P = {p.name: p for p in L.parts}
    if side == "L":
        R, c = NECK_ARMOR
        R = np.array(R, float)
        L.add("armor_neck", "armor, neck", P["neck"].matrix @ place(R, c, F["neck.ball_mid"]), "neck")
    for stl, (host, R, t) in ARMOR.items():
        is_limb = host.endswith(".L")
        if not is_limb and side != "L":
            continue
        hp = P[host.replace(".L", f".{side}")] if is_limb else P[host]
        R = np.array(R, float)
        if t is None:  # 膝のカバー
            a, b = KNEE_ARMOR_CLIP
            t = np.array(b) - R @ np.array(a)
        rel = mat(R, t)
        name = "armor_" + stl.replace("armor, ", "").replace(", 2x", "").replace(" ", "_") + (f".{side}" if is_limb else "")
        bone = hp.bone
        if stl == "armor, foot, 2x":
            bone = f"foot.{side}"
        M = L.add(name, stl, hp.matrix @ rel, bone, ARMOR_SYMMETRY.get(stl))
        if stl == "armor, foot, 2x":  # つま先は足のアーマーに付ける
            Rt, a, b = TOE
            Rt = np.array(Rt, float)
            L.add(f"armor_toe.{side}", "armor, toe, 2x", M @ place(Rt, a, b), bone, ARMOR_SYMMETRY["armor, toe, 2x"])
            L.joints[f"toe.{side}"] = apply(M @ place(Rt, a, b), TOE_TIP)


def build(hands: str = "open", armor: bool = True) -> Layout:
    """T ポーズの配置を計算する。hands は手の STL（open / fist / grip）、armor=False でフレームだけ。"""
    L = Layout()
    J = L.joints
    R_torso = basis(x=(1, 0, 0), y=(0, 0, 1))  # ローカル Z → ワールド -Y（正面）

    # --- 胴体
    hips = L.add("hips", STL["hips"], place(R_torso, F["hips.center_ball"], (0, 0, 0)), "hips")
    J["hips"] = apply(hips, F["hips.center_ball"])
    J["hips_tail"] = J["hips"] + (0, 0, -6)
    # waist のソケットは左右に伸びる hips の棒を横から挟むので、体の側面と平行な向き（ガイド手順5）。
    # head は頭のアーマー内側の空洞（奥行き 5.3 × 幅 8.5）に正面向きでしか入らないので R_torso のまま。
    R_side = basis(x=(0, -1, 0), y=(0, 0, 1))  # ローカル Z → ワールド X
    waist = L.add("waist", STL["waist"], place(R_side, F["waist.socket"], J["hips"]), "waist")
    J["waist_top"] = apply(waist, F["waist.ball"])
    abdomen = L.add("abdomen", STL["abdomen"], place(R_torso, F["abdomen.socket"], J["waist_top"]), "abdomen")
    J["abdomen_top"] = apply(abdomen, F["abdomen.ball"])
    chest = L.add("chest", STL["chest"], place(R_torso, F["chest.socket"], J["abdomen_top"]), "chest")
    J["neck_base"] = apply(chest, F["chest.neck_socket"])
    neck = L.add("neck", STL["neck"], place(R_torso, F["neck.ball_low"], J["neck_base"]), "neck")
    J["neck_mid"] = apply(neck, F["neck.ball_mid"])
    J["neck_top"] = apply(neck, F["neck.ball_top"])
    head = L.add("head", STL["head"], place(R_torso, F["head.socket"], J["neck_top"]), "head")
    J["head_top"] = apply(head, (0, 11.2 - 2.2, 2.5))

    side = "L"  # 左半身だけ組み、右半身は最後に鏡映で作る

    # --- 腕（T ポーズ、左腕は +X 方向に伸びる。肘は前（-Y）に曲がる）
    J[f"clavicle_base.{side}"] = apply(chest, F[f"chest.ball_{side}"])
    R_clav = basis(y=(1, 0, 0), z=(0, -1, 0))
    clav = L.add(f"clavicle.{side}", STL["clavicle"],
                 place(R_clav, F["clavicle.socket"], J[f"clavicle_base.{side}"]), f"clavicle.{side}",
                 sym=SYMMETRY["clavicle"])
    J[f"shoulder.{side}"] = apply(clav, F["clavicle.ball"])
    R_arm = basis(y=(-1, 0, 0), z=(0, 0, 1))  # ローカル Y が肩の方、Z がヒンジ軸（上下）
    hs = L.add(f"shoulder_joint.{side}", STL["hs"], place(R_arm, F["hs.socket"], J[f"shoulder.{side}"]),
               f"upper_arm.{side}", sym=SYMMETRY["hs"])
    ua = L.add(f"upper_arm.{side}", STL["upper_arm"], hs @ mat(np.eye(3), -np.array(F["upper_arm.hs"])),
               f"upper_arm.{side}", sym=SYMMETRY["upper_arm"])
    J[f"elbow1.{side}"] = apply(ua, F["upper_arm.pin"])
    R_elbow = basis(x=(1, 0, 0), z=(0, 0, 1))  # clip2 が手先側、アーマー用ピンが後ろ（+Y）
    el = L.add(f"elbow.{side}", STL["hinge"], place(R_elbow, F["hinge.clip1"], J[f"elbow1.{side}"]),
               f"elbow.{side}", sym=SYMMETRY["hinge"])
    J[f"elbow2.{side}"] = apply(el, F["hinge.clip2"])
    fa = L.add(f"forearm.{side}", STL["forearm"], place(R_arm, F["forearm.pin"], J[f"elbow2.{side}"]),
               f"forearm.{side}", sym=SYMMETRY["forearm"])
    J[f"wrist.{side}"] = apply(fa, F["forearm.ball"])
    R_hand = basis(y=(1, 0, 0), z=(0, 1, 0))  # 指先が外、手のひらが後ろ、親指が下（実物で確認）
    hd = L.add(f"hand.{side}", STL["hand"].format(pose=hands), place(R_hand, F["hand.socket"], J[f"wrist.{side}"]),
               f"hand.{side}", sym=SYMMETRY["hand"])
    J[f"hand_tip.{side}"] = apply(hd, (0, 9, 0))

    # --- 脚（膝は後ろ（+Y）に曲がる）
    J[f"hip.{side}"] = apply(hips, F[f"hips.ball_{side}"])
    R_leg = basis(y=(0, 0, 1), z=(1, 0, 0))  # ローカル Y が上、Z がヒンジ軸（左右）
    hs = L.add(f"hip_joint.{side}", STL["hs"], place(R_leg, F["hs.socket"], J[f"hip.{side}"]), f"thigh.{side}",
               sym=SYMMETRY["hs"])
    th = L.add(f"thigh.{side}", STL["thigh"], hs @ mat(np.eye(3), -np.array(F["thigh.hs"])), f"thigh.{side}",
               sym=SYMMETRY["thigh"])
    J[f"knee1.{side}"] = apply(th, F["thigh.pin"])
    R_knee = basis(x=(0, 0, -1), z=(-1, 0, 0))  # clip2 が下、アーマー用ピンが前（-Y）
    kn = L.add(f"knee.{side}", STL["hinge"], place(R_knee, F["hinge.clip1"], J[f"knee1.{side}"]),
               f"knee.{side}", sym=SYMMETRY["hinge"])
    J[f"knee2.{side}"] = apply(kn, F["hinge.clip2"])
    sh = L.add(f"shin.{side}", STL["shin"], place(R_leg, F["shin.pin"], J[f"knee2.{side}"]), f"shin.{side}",
               sym=SYMMETRY["shin"])
    J[f"ankle.{side}"] = apply(sh, F["shin.ball"])
    # 足首パーツは正面から C の形が見える向き（ガイド手順15）。下のピンは前後方向の軸になる
    R_ankle = basis(y=(0, 0, 1), z=(0, 1, 0))  # C の高い側の翼が外側（左右で鏡写し。実物で確認）
    an = L.add(f"ankle.{side}", STL["ankle"], place(R_ankle, F["ankle.socket"], J[f"ankle.{side}"]),
               f"ankle.{side}", sym=SYMMETRY["ankle"])
    J[f"foot_pin.{side}"] = apply(an, F["ankle.pin"])
    J[f"toe.{side}"] = J[f"foot_pin.{side}"] + (0, -14, -2)  # アーマー無しのときの仮の値

    if armor:
        add_armor(L, side)

    # --- 右半身 = 左半身の鏡映
    for p in [p for p in L.parts if p.name.endswith(".L")]:
        M = WORLD_MIRROR @ p.matrix @ p.sym
        L.add(p.name[:-2] + ".R", p.stl.replace(", left", ", right"), M, p.bone[:-2] + ".R", p.sym)
    for n in [n for n in J if n.endswith(".L")]:
        J[n[:-2] + ".R"] = J[n] * np.array([-1, 1, 1])
    return L


if __name__ == "__main__":
    lay = build()
    for n, p in lay.joints.items():
        print(f"{n:18s} {np.round(p, 2)}")
