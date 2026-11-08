"""Blender で Dummy13 の合成データを生成する。

リグ入りの .blend を開き、ポーズ・カメラ・照明・材質をランダム化してレンダリングし、
YOLO-pose 形式のラベルとメタデータ（関節角度・3D座標）を自動で書き出す。
背景は透過（RGBA）でレンダリングし、背景合成は後段の composite.py で行う。

使い方（Blender 本体）:
    blender -b models/dummy13_rig.blend -P synth/generate.py -- --count 100

使い方（pip の bpy モジュール）:
    python synth/generate.py --blend models/dummy13_rig.blend --count 100

出力（既定 data/synth/raw/）:
    images/000000.png   RGBA 画像
    labels/000000.txt   YOLO-pose ラベル（class cx cy w h x y v ...、正規化座標）
    meta/000000.json    関節角度・カメラ座標系の3Dキーポイントなど
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Euler, Quaternion, Vector
from mathutils.bvhtree import BVHTree

REPO = Path(__file__).resolve().parent.parent
PREFIX = "SYN_"  # このスクリプトが作るオブジェクトの接頭辞


# ---------------------------------------------------------------- 設定読み込み


def parse_args() -> argparse.Namespace:
    # Blender 経由では "--" 以降がスクリプト引数
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    elif "-P" in sys.argv or "--python" in sys.argv:  # blender -b -P script.py（スクリプト用の引数なし）
        argv = []
    else:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description="Dummy13 合成データ生成")
    p.add_argument("--blend", help="開く .blend（Blender 本体で -b 指定済みなら不要）")
    p.add_argument("--config", default=str(REPO / "synth" / "rig_config.json"))
    p.add_argument("--keypoints", default=str(REPO / "configs" / "keypoints.yaml"))
    p.add_argument("--out", default=str(REPO / "data" / "synth" / "raw"))
    p.add_argument("--count", type=int, default=10, help="生成枚数")
    p.add_argument("--start", type=int, default=0, help="開始番号（並列実行で分割するとき用）")
    p.add_argument("--seed", type=int, default=0, help="乱数シード（番号ごとに seed+番号 を使う）")
    p.add_argument("--dry-run", action="store_true", help="レンダリングせずラベルだけ書き出す")
    return p.parse_args(argv)


def load_keypoint_names(path: str) -> list[str]:
    """configs/keypoints.yaml の keypoints リストだけを読む。

    Blender 同梱の Python には PyYAML が無いことが多いので簡易パーサで読む。
    """
    names, in_list = [], False
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.split("#", 1)[0].rstrip()
        if not s:
            continue
        if not line[0].isspace():
            in_list = s == "keypoints:"
            continue
        if in_list and s.strip().startswith("- "):
            names.append(s.strip()[2:].strip())
    if not names:
        raise ValueError(f"{path} に keypoints が見つかりません")
    return names


# ---------------------------------------------------------------- シーン準備


def find_figure_meshes(arm: bpy.types.Object, names: list[str]) -> list[bpy.types.Object]:
    """フィギュア本体のメッシュを集める。

    names 指定が無ければ、アーマチュアの子孫（分割パーツをボーンに親子付けした場合）と、
    アーマチュアモディファイアでこのアーマチュアを使うメッシュ（一体モデルをスキニングした場合）。
    """
    if names:
        return [bpy.data.objects[n] for n in names]
    meshes = []
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.name.startswith(PREFIX):
            continue
        parent, is_child = obj.parent, False
        while parent is not None:
            if parent == arm:
                is_child = True
                break
            parent = parent.parent
        skinned = any(m.type == "ARMATURE" and m.object == arm for m in obj.modifiers)
        if is_child or skinned:
            meshes.append(obj)
    if not meshes:
        raise RuntimeError("フィギュアのメッシュが見つかりません（rig_config.json の figure_objects で指定してください）")
    return meshes


def validate_config(cfg: dict, arm: bpy.types.Object, kp_names: list[str]) -> None:
    errors = []
    for name in kp_names:
        spec = cfg["keypoints"].get(name)
        if spec is None:
            errors.append(f"keypoints.{name} が rig_config.json にありません")
        elif "bone" in spec and spec["bone"] not in arm.pose.bones:
            errors.append(f"keypoints.{name}: ボーン '{spec['bone']}' がアーマチュアにありません")
        elif "object" in spec and spec["object"] not in bpy.data.objects:
            errors.append(f"keypoints.{name}: オブジェクト '{spec['object']}' がありません")
    for bone in cfg["joint_limits_deg"]:
        if bone.startswith("_"):
            continue
        if bone not in arm.pose.bones:
            errors.append(f"joint_limits_deg: ボーン '{bone}' がアーマチュアにありません")
    if errors:
        bones = ", ".join(b.name for b in arm.pose.bones)
        raise RuntimeError("設定エラー:\n  " + "\n  ".join(errors) + f"\nアーマチュアのボーン: {bones}")


def enable_cycles_gpu(scene: bpy.types.Scene) -> None:
    """Cycles で GPU（OptiX → CUDA → HIP → Metal → oneAPI の順に探す）を使う。無ければ CPU のまま。"""
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
        try:
            prefs.compute_device_type = kind
        except TypeError:
            continue
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == kind]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == kind
            scene.cycles.device = "GPU"
            print(f"[synth] Cycles: GPU を使用（{kind}: {', '.join(d.name for d in gpus)}）")
            return
    print("[synth] Cycles: GPU が見つからないので CPU で描画します")


def setup_render(scene: bpy.types.Scene, rcfg: dict) -> None:
    engine = rcfg.get("engine", "EEVEE").upper()
    candidates = ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"] if engine == "EEVEE" else [engine]
    for c in candidates:  # Blender 4.2〜4.4 は EEVEE_NEXT、それ以外は EEVEE
        try:
            scene.render.engine = c
            break
        except TypeError:
            continue
    samples = rcfg.get("samples", 16)
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        enable_cycles_gpu(scene)
    else:
        scene.eevee.taa_render_samples = samples
    w, h = rcfg.get("resolution", [640, 480])
    scene.render.resolution_x, scene.render.resolution_y = w, h
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"


def get_or_create(name: str, kind: str, data_type: str | None = None) -> bpy.types.Object:
    obj = bpy.data.objects.get(name)
    if obj is not None:
        return obj
    if kind == "CAMERA":
        data = bpy.data.cameras.new(name)
    else:
        data = bpy.data.lights.new(name, type=data_type)
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def setup_world(scene: bpy.types.Scene) -> bpy.types.ShaderNode:
    if scene.world is None:
        scene.world = bpy.data.worlds.new(PREFIX + "World")
    world = scene.world
    if bpy.app.version < (5, 0, 0):  # 5.0 以降は常にノード
        world.use_nodes = True
    bg = next((n for n in world.node_tree.nodes if n.type == "BACKGROUND"), None)
    if bg is None:
        bg = world.node_tree.nodes.new("ShaderNodeBackground")
        out = next((n for n in world.node_tree.nodes if n.type == "OUTPUT_WORLD"), None)
        if out is None:
            out = world.node_tree.nodes.new("ShaderNodeOutputWorld")
        world.node_tree.links.new(bg.outputs[0], out.inputs[0])
    return bg


def make_materials(n: int) -> list[bpy.types.Material]:
    mats = []
    for i in range(n):
        name = f"{PREFIX}Mat_{i}"
        mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
        if bpy.app.version < (5, 0, 0):
            mat.use_nodes = True
        mats.append(mat)
    return mats


# ---------------------------------------------------------------- ランダム化


def randomize_pose(arm, limits: dict, neutral_prob: float, rng: random.Random, gamma: float = 1.0,
                   center: dict | None = None, bend: dict | None = None, keypoints: dict | None = None,
                   checker=None) -> dict:
    """joint_limits_deg の範囲で各ボーンの回転をランダムに決める。戻り値は角度（度）。

    gamma > 1 で center（既定は T ポーズ＝0°）寄りの角度が出やすくなる
    （全関節を同時に目いっぱい曲げるとめり込みやすいため）。center で「腕を下ろした姿勢」などを中心にできる。
    """
    for pb in arm.pose.bones:  # 毎回レストポーズから始める
        pb.location = (0, 0, 0)
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.rotation_euler = (0, 0, 0)
        pb.scale = (1, 1, 1)
    # 一定確率で「気をつけ」寄りの控えめなポーズにする（実際によく取る姿勢を厚めに）
    shrink = 0.3 if rng.random() < neutral_prob else 1.0
    angles = {}
    for bone, lim in limits.items():
        if bone.startswith("_"):
            continue
        pb = arm.pose.bones[bone]
        pb.rotation_mode = "XYZ"
        deg = []
        for i, axis in enumerate("xyz"):
            lo, hi = lim.get(axis, [0, 0])
            c = min(max((center or {}).get(bone, [0, 0, 0])[i], lo), hi)
            u = rng.uniform(-1, 1)
            v = c + abs(u) ** gamma * shrink * ((hi - c) if u > 0 else (lo - c))
            deg.append(v)
        pb.rotation_euler = Euler([math.radians(v) for v in deg], "XYZ")
        angles[bone] = deg
    # 肘・膝（二重ヒンジ）は、上の出し方だとほとんど伸びたままになるので、
    # 手足ごとに bend_prob の確率で「曲げる向き」に一様に曲げ直す
    # prob は 1 つの値か、{"arm": .., "leg": ..} のように手足の種類ごと
    groups = {}
    for bone in (bend or {}).get("deg", {}):
        kind = "arm" if bone.startswith(("elbow", "forearm")) else "leg"
        groups.setdefault((kind, bone.split(".")[-1]), []).append(bone)
    for (kind, _), bones in groups.items():
        prob = bend.get("prob", 0.0)
        if rng.random() >= (prob.get(kind, 0.0) if isinstance(prob, dict) else prob):
            continue
        for bone in bones:
            lo, hi = bend["deg"][bone]
            angles[bone][0] = rng.uniform(lo, hi)
            arm.pose.bones[bone].rotation_euler = Euler([math.radians(v) for v in angles[bone]], "XYZ")
    # 上の出し方ではほぼ出ない形を、ボーンごとに prob の確率で、指定した軸だけ一様に出し直す
    #   raise: 腕を上げる（肘を曲げると前腕が上を向く「力こぶ」の形など）
    #   reach: 腕を前に出す（肩から前へ上げる。肘の曲がる向きはねじりで変わる）
    #   open:  腿を横に開いて大きくひねる（すねが下を向く）
    #   walk:  歩く・走る（右腕と左脚、左腕と右脚を前後逆に振る）
    #   sit:   椅子に座る（体はまっすぐ、両腿を前に・両膝を曲げる）
    #   kneel: 片膝をつく（片脚は腿を前に、もう片脚は腿を下に向けてすねを後ろへ）
    #   arch:  体を後ろに反らせる（腰・お腹・胸・首・頭をまとめて。腕を上げる形は raise と重なって出る）
    #   crouch: 前かがみでしゃがむ（体・両腿・両膝をまとめて。together=true なら全ボーンで 1 回だけ抽選）
    # mirror=true なら半分の確率で左右を入れ替える（.L と .R を入れ替え、y・z の向きを反転）
    for mode in ("raise", "reach", "kneeup", "open", "walk", "sit", "kneel", "arch", "crouch"):
        m = (bend or {}).get(mode, {})
        hit = rng.random() < m.get("prob", 0.0)
        deg = m.get("deg", {})
        if m.get("mirror") and rng.random() < 0.5:
            deg = {mirror_name(b): {a: (sorted((-lo, -hi)) if a in "yz" else (lo, hi)) for a, (lo, hi) in axes.items()}
                   for b, axes in deg.items()}
        for bone, axes in deg.items():
            if not (hit if m.get("together") else rng.random() < m.get("prob", 0.0)):
                continue
            for axis, (lo, hi) in axes.items():
                angles[bone]["xyz".index(axis)] = rng.uniform(lo, hi)
            arm.pose.bones[bone].rotation_euler = Euler([math.radians(v) for v in angles[bone]], "XYZ")
    touch = (bend or {}).get("touch")
    if touch and keypoints:
        touch_targets(arm, limits, angles, touch, keypoints, rng, checker)
    return angles


def mirror_name(bone: str) -> str:
    """upper_arm.L ⇔ upper_arm.R"""
    return bone[:-2] + {".L": ".R", ".R": ".L"}[bone[-2:]] if bone[-2:] in (".L", ".R") else bone


def touch_targets(arm, limits: dict, angles: dict, touch: dict, keypoints: dict, rng: random.Random,
                  checker=None) -> None:
    """腕ごとに確率で、手首を体のどこか（顔の前・腰・後頭部・背中・胸の前）に持っていく。

    角度では指定しにくいので、腕の角度の候補をいくつも作り、手首が目標に一番近いものを選ぶ。
    targets の各項目: at = 基準のキーポイント（{s} は自分の側の l/r）、offset = そこからのずれ（m）。
    座標は Dummy13 の前が -Y、左が +X（リグを置いたままの向き）。offset の x は「外側が +」で、右腕では反転する。
    both = その目標になったとき、もう片方の腕も同じ目標にする確率（両手を頭の後ろ・腕組みなど）。
    checker があれば、めり込まない候補だけを選ぶ（体に手を寄せるのでめり込みやすく、ポーズ全体の引き直しが増えるため）。
    """
    targets = touch.get("targets", {})
    chosen = {}
    for side in ("L", "R"):
        if side in chosen:
            continue
        r, acc = rng.random(), 0.0
        for name, t in targets.items():
            acc += t.get("prob", 0.0)
            if r < acc:
                chosen[side] = name
                other = "R" if side == "L" else "L"
                if other not in chosen and rng.random() < t.get("both", 0.0):
                    chosen[other] = name
                break
    if not chosen:
        return
    bpy.context.view_layer.update()
    base = {}  # 目標は腕を動かす前の体で決める（腕組みで先に動かした腕に引きずられないように）
    for side, name in chosen.items():
        t = targets[name]
        sx = 1.0 if side == "L" else -1.0
        kp = t["at"].replace("{s}", "l" if side == "L" else "r")
        ox, oy, oz = t.get("offset", [0, 0, 0])
        base[side] = keypoint_world(arm, keypoints[kp]) + Vector((sx * ox, oy, oz))
    if checker is not None:
        # 手を寄せる腕を下ろした状態でもめり込むなら、このポーズはどのみち引き直しになるので検査を省く
        rest = {}
        for side in base:
            rest.update({f"upper_arm.{side}": [0, 0, -60 if side == "L" else 60],
                         f"elbow.{side}": [0, 0, 0], f"forearm.{side}": [0, 0, 0]})
        set_angles(arm, rest)
        if checker.collides() is not None:
            checker = None
    for side, target in base.items():
        bones = [f"upper_arm.{side}", f"elbow.{side}", f"forearm.{side}"]
        cands = []
        for _ in range(touch.get("tries", 48)):
            cand = {}
            for b in bones:
                lim = limits[b]
                cand[b] = [rng.uniform(*lim.get(a, [0, 0])) for a in "xyz"]
            cand[f"forearm.{side}"][0] = rng.uniform(60, lim.get("x", [0, 110])[1])  # 手を体に寄せるので肘は深めに
            set_angles(arm, cand)
            d = (keypoint_world(arm, keypoints[f"{'l' if side == 'L' else 'r'}_wrist"]) - target).length
            cands.append((d, cand))
        # めり込まない一番近い候補から始め、そのまわりを少しずつ動かして詰める
        # （ランダムな候補だけだと頭の後ろ・背中などに届きにくい）。めり込みの検査は重いので、
        # 近づいたときだけ・合計 check 回までにする。checker が無ければ検査しない
        budget = touch.get("check", 0) if checker is not None else 0

        def free(cand):
            nonlocal budget
            if checker is None:
                return True
            if budget <= 0:
                return False
            budget -= 1
            set_angles(arm, cand)
            return checker.collides() is None

        cands.sort(key=lambda c: c[0])
        best_d, best = cands[0]
        for d, cand in cands[:max(budget // 2, 1)]:
            if free(cand):
                best_d, best = d, cand
                break
        steps = touch.get("refine", 0)
        for i in range(steps):
            sd = 20.0 * (1 - i / steps) + 2.0
            cand = {b: [min(max(v + rng.gauss(0, sd), limits[b].get(a, [0, 0])[0]), limits[b].get(a, [0, 0])[1])
                        for v, a in zip(deg, "xyz")] for b, deg in best.items()}
            set_angles(arm, cand)
            d = (keypoint_world(arm, keypoints[f"{'l' if side == 'L' else 'r'}_wrist"]) - target).length
            if d < best_d - 0.001 and free(cand):
                best_d, best = d, cand
        set_angles(arm, best)
        angles.update(best)


def set_angles(arm, angles: dict) -> None:
    for b, deg in angles.items():
        arm.pose.bones[b].rotation_euler = Euler([math.radians(v) for v in deg], "XYZ")
    bpy.context.view_layer.update()


def randomize_materials(meshes, mats, mcfg: dict, rng: random.Random) -> None:
    """3Dプリントのフィラメント色を想定して、パーツごとに色・粗さを変える。"""
    n = rng.randint(1, len(mats))
    for mat in mats[:n]:
        bsdf = next((nd for nd in mat.node_tree.nodes if nd.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            continue
        h, s = rng.random(), rng.random() ** 0.7
        v = rng.uniform(*mcfg.get("value", [0.05, 0.95]))
        r, g, b = _hsv_to_rgb(h, s, v)
        bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
        bsdf.inputs["Roughness"].default_value = rng.uniform(*mcfg.get("roughness", [0.25, 0.8]))
    per_object = mcfg.get("per_object_colors", True)
    shared = rng.choice(mats[:n])
    # 実物に多い「フレームとアーマーで2色」の塗り分け（オブジェクト名が armor_ で始まるものがアーマー）
    two_tone = n >= 2 and rng.random() < mcfg.get("two_tone_prob", 0.0)
    for obj in meshes:
        if two_tone:
            mat = mats[0] if obj.name.startswith("armor_") else mats[1]
        else:
            mat = rng.choice(mats[:n]) if per_object else shared
        if not obj.material_slots:
            obj.data.materials.append(mat)
        for slot in obj.material_slots:
            slot.material = mat


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]


def direction_from_angles(az: float, el: float) -> Vector:
    return Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))


def look_at(obj, target: Vector, roll: float = 0.0) -> None:
    rot = (target - obj.location).to_track_quat("-Z", "Y")
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = rot @ Quaternion((0, 0, 1), roll)


def place_camera(cam, center: Vector, size: float, ccfg: dict, rng: random.Random) -> dict:
    cam.data.sensor_fit = "AUTO"
    cam.data.sensor_width = 36.0
    cam.data.lens = rng.uniform(*ccfg.get("focal_mm", [20, 60]))
    bpy.context.view_layer.update()
    fov = min(cam.data.angle_x, cam.data.angle_y)
    fill = rng.uniform(*ccfg.get("fill", [1.2, 3.5]))  # 大きいほどフィギュアが小さく写る
    dist = (size / 2) / math.tan(fov / 2) * fill
    az = rng.uniform(0, 2 * math.pi)
    # 手に持って後ろへ反らす・前へ倒すと、フィギュアから見てカメラが下や上に回るので広めに取る
    el = math.radians(rng.uniform(*ccfg.get("elevation_deg", [-60, 80])))
    jitter = ccfg.get("target_jitter", 0.15) * size
    target = center + Vector([rng.uniform(-jitter, jitter) for _ in range(3)])
    cam.location = target + direction_from_angles(az, el) * dist
    # 机に寝かせて上から撮る・手で持って傾けるなど、画面内でフィギュアが大きく回ることもあるので
    # 一定の確率で全方向に回す
    if rng.random() < ccfg.get("roll_full_prob", 0.0):
        roll = math.radians(rng.uniform(-180, 180))
    else:
        roll = math.radians(rng.uniform(*ccfg.get("roll_deg", [-15, 15])))
    look_at(cam, target, roll)
    cam.data.clip_start = dist * 0.01
    cam.data.clip_end = dist * 10
    return {"focal_mm": cam.data.lens, "sensor_mm": 36.0, "distance": dist,
            "azimuth_deg": math.degrees(az), "elevation_deg": math.degrees(el),
            "roll_deg": math.degrees(roll)}


class HdriWorld:
    """ワールドの照明に屋内 HDRI（synth/download_backgrounds.py で取得）をランダムに使う。

    背景は透過でレンダリングするので HDRI 自体は写らず、映り込みと光の当たり方だけが実際の部屋に近づく。
    """

    def __init__(self, scene, bg_node, files: list, cache: int = 8):
        self.files, self.cache, self.loaded = files, cache, []
        nt = scene.world.node_tree
        self.bg = bg_node
        self.env = nt.nodes.new("ShaderNodeTexEnvironment")
        coord = nt.nodes.new("ShaderNodeTexCoord")
        self.mapping = nt.nodes.new("ShaderNodeMapping")
        nt.links.new(coord.outputs["Generated"], self.mapping.inputs["Vector"])
        nt.links.new(self.mapping.outputs["Vector"], self.env.inputs["Vector"])
        self.nt = nt

    def use(self, enable: bool, rng: random.Random, strength: float) -> None:
        links = self.bg.inputs["Color"].links
        if not enable:
            for link in links:
                self.nt.links.remove(link)
            return
        path = rng.choice(self.files)
        img = bpy.data.images.get(path.name) or bpy.data.images.load(str(path))
        if img not in self.loaded:
            self.loaded.append(img)
            if len(self.loaded) > self.cache:  # メモリを食うので古いものから捨てる
                old = self.loaded.pop(0)
                if old != img:
                    bpy.data.images.remove(old)
        self.env.image = img
        self.mapping.inputs["Rotation"].default_value = (0, 0, rng.uniform(0, 2 * math.pi))
        if not links:
            self.nt.links.new(self.env.outputs["Color"], self.bg.inputs["Color"])
        self.bg.inputs["Strength"].default_value = strength


def randomize_lights(lights, world_bg, center: Vector, lcfg: dict, rng: random.Random, hdri=None) -> None:
    use_hdri = hdri is not None and rng.random() < lcfg.get("hdri_prob", 0.7)
    sun_scale = rng.uniform(0.0, 0.6) if use_hdri else 1.0  # HDRI のときは太陽光を弱める（無しもあり）
    for i, light in enumerate(lights):
        az = rng.uniform(0, 2 * math.pi)
        el = math.radians(rng.uniform(10, 85))
        light.location = center + direction_from_angles(az, el)
        look_at(light, center)
        key = lcfg.get("key_strength" if i == 0 else "fill_strength", [1.0, 5.0] if i == 0 else [0.0, 1.5])
        light.data.energy = rng.uniform(*key) * sun_scale
        light.data.color = [rng.uniform(0.8, 1.0) for _ in range(3)]
        light.data.angle = math.radians(rng.uniform(0.5, 20))  # 影の柔らかさ
    if hdri is not None:
        hdri.use(use_hdri, rng, rng.uniform(*lcfg.get("hdri_strength", [0.5, 2.0])))
    if not use_hdri:
        world_bg.inputs["Color"].default_value = (*[rng.uniform(0.3, 1.0) for _ in range(3)], 1.0)
        world_bg.inputs["Strength"].default_value = rng.uniform(*lcfg.get("ambient_strength", [0.1, 1.0]))


# ---------------------------------------------------------------- めり込み検査


class CollisionChecker:
    """ポーズ全体で部品どうしがめり込んでいないかを調べる。

    隣り合うボーン（同じボーン・親子）に付いた部品どうしは関節機構でつながっているので調べない。
    それ以外の組について、片方の頂点（間引いたもの）がもう片方のメッシュの内側に depth 以上入り込んだ数を数え、
    T ポーズ時より max_new 個を超えて増えたらめり込みとみなす（estimate_limits.py と同じ考え方）。
    """

    def __init__(self, arm, meshes, depth: float, max_new: int, n_points: int = 300):
        self.arm, self.depth, self.max_new = arm, depth, max_new
        self.objs = [o for o in meshes if o.parent == arm and o.parent_type == "BONE" and o.parent_bone]
        self.tris, self.sample = {}, {}
        for o in self.objs:
            o.data.calc_loop_triangles()
            t = np.empty(len(o.data.loop_triangles) * 3, dtype=np.int32)
            o.data.loop_triangles.foreach_get("vertices", t)
            self.tris[o.name] = t.reshape(-1, 3).tolist()
            n = len(o.data.vertices)
            self.sample[o.name] = np.linspace(0, n - 1, min(n, n_points)).astype(int)
        bones = arm.data.bones

        def adjacent(a, b):
            a, b = bones[a], bones[b]
            return a == b or a.parent == b or b.parent == a
        self.pairs = [(a, b) for i, a in enumerate(self.objs) for b in self.objs[i + 1:]
                      if not adjacent(a.parent_bone, b.parent_bone)]
        self.base = self._count(rest=True)

    def _count(self, rest=False) -> dict:
        if rest:
            saved = self.arm.data.pose_position
            self.arm.data.pose_position = "REST"
        bpy.context.view_layer.update()
        world, box = {}, {}
        for o in self.objs:
            co = np.empty(len(o.data.vertices) * 3)
            o.data.vertices.foreach_get("co", co)
            mw = np.array(o.matrix_world)
            w = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
            world[o.name], box[o.name] = w, (w.min(0), w.max(0))
        trees, counts = {}, {}

        def tree(o):
            if o.name not in trees:
                trees[o.name] = BVHTree.FromPolygons(world[o.name].tolist(), self.tris[o.name])
            return trees[o.name]

        def inside(src, dst):  # src の頂点のうち dst の内側に depth 以上入っている数
            lo, hi = box[dst.name]
            pts = world[src.name][self.sample[src.name]]
            pts = pts[np.all((pts > lo) & (pts < hi), axis=1)]
            t, n = tree(dst), 0
            for p in pts:
                loc, nor, _, dist = t.find_nearest(Vector(p))
                if loc is not None and dist > self.depth and (Vector(p) - loc).dot(nor) < 0:
                    n += 1
            return n

        for a, b in self.pairs:
            (alo, ahi), (blo, bhi) = box[a.name], box[b.name]
            if np.any(alo > bhi) or np.any(blo > ahi):
                continue
            n = inside(a, b) + inside(b, a)
            if n:
                counts[(a.name, b.name)] = n
        if rest:
            self.arm.data.pose_position = saved
            bpy.context.view_layer.update()
        return counts

    def collides(self):
        """めり込んでいる部品の組（無ければ None）。"""
        for key, n in self._count().items():
            if n > self.base.get(key, 0) + self.max_new:
                return key
        return None


# ---------------------------------------------------------------- 投影・ラベル


def figure_vertices(meshes, depsgraph) -> np.ndarray:
    """ポーズ適用後のメッシュ頂点（ワールド座標）。"""
    chunks = []
    for obj in meshes:
        ev = obj.evaluated_get(depsgraph)
        mesh = ev.to_mesh()
        co = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", co)
        mw = np.array(ev.matrix_world)
        chunks.append(co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3])
        ev.to_mesh_clear()
    return np.concatenate(chunks)


def project(scene, cam, pts: np.ndarray) -> np.ndarray:
    """ワールド座標 → (x, y, depth)。x, y は画像左上原点の正規化座標。

    bpy_extras.object_utils.world_to_camera_view を numpy でまとめて計算したもの。
    """
    inv = np.array(cam.matrix_world.normalized().inverted())
    local = pts @ inv[:3, :3].T + inv[:3, 3]
    z = -local[:, 2]
    frame = cam.data.view_frame(scene=scene)[:3]  # 右上, 右下, 左下（z は負）
    fz = -frame[0].z
    min_x, max_x = frame[2].x, frame[1].x
    min_y, max_y = frame[1].y, frame[0].y
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(z > 0, fz / z, 0.0)
    x = (local[:, 0] * s - min_x) / (max_x - min_x)
    y = (local[:, 1] * s - min_y) / (max_y - min_y)
    return np.stack([x, 1.0 - y, z], axis=1)


def keypoint_world(arm, spec: dict) -> Vector:
    if "object" in spec:
        return bpy.data.objects[spec["object"]].matrix_world.translation.copy()
    pb = arm.pose.bones[spec["bone"]]
    at = spec.get("at", "head")
    t = {"head": 0.0, "tail": 1.0}.get(at, at)
    return arm.matrix_world @ pb.head.lerp(pb.tail, float(t))


def visibility(scene, depsgraph, cam_loc: Vector, p: Vector, uv, tol: float) -> int:
    """YOLO 形式の可視フラグ。0=画面外, 1=隠れている, 2=見えている。

    関節点はメッシュの内側にあるので、手前のメッシュ表面との距離が tol 以内なら「見えている」とみなす。
    """
    x, y, z = uv
    if z <= 0 or not (0 <= x <= 1 and 0 <= y <= 1):
        return 0
    d = p - cam_loc
    dist = d.length
    hit, loc, *_ = scene.ray_cast(depsgraph, cam_loc, d.normalized(), distance=dist)
    if hit and dist - (loc - cam_loc).length > tol:
        return 1
    return 2


def camera_space(cam, p: Vector) -> list[float]:
    """カメラ座標（x 右, y 下, z 奥; OpenCV と同じ向き）。"""
    c = cam.matrix_world.normalized().inverted() @ p
    return [c.x, -c.y, -c.z]


# ---------------------------------------------------------------- メイン


def main() -> None:
    args = parse_args()
    if args.blend:
        bpy.ops.wm.open_mainfile(filepath=str(Path(args.blend).resolve()))
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    kp_names = load_keypoint_names(args.keypoints)

    scene = bpy.context.scene
    arm = bpy.data.objects.get(cfg["armature"])
    if arm is None or arm.type != "ARMATURE":
        arms = [o.name for o in bpy.data.objects if o.type == "ARMATURE"]
        raise RuntimeError(f"アーマチュア '{cfg['armature']}' がありません（候補: {arms}）")
    validate_config(cfg, arm, kp_names)
    meshes = find_figure_meshes(arm, cfg.get("figure_objects", []))
    print(f"[synth] アーマチュア: {arm.name} / メッシュ: {[m.name for m in meshes]}")

    setup_render(scene, cfg.get("render", {}))
    cam = get_or_create(PREFIX + "Camera", "CAMERA")
    scene.camera = cam
    lights = [get_or_create(PREFIX + n, "LIGHT", "SUN") for n in ("Key", "Fill")]
    world_bg = setup_world(scene)
    lcfg = cfg.get("light", {})
    hdri_dir = REPO / lcfg.get("hdri_dir", "data/hdri")
    hdri_files = sorted(hdri_dir.glob("*.hdr")) + sorted(hdri_dir.glob("*.exr")) if hdri_dir.is_dir() else []
    hdri = HdriWorld(scene, world_bg, hdri_files) if hdri_files else None
    print(f"[synth] 照明用 HDRI: {len(hdri_files)} 件（{hdri_dir}）")
    mcfg = cfg.get("material", {})
    mats = make_materials(mcfg.get("max_colors", 3)) if mcfg.get("randomize", True) else []

    out = Path(args.out)
    for sub in ("images", "labels", "meta"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    ccfg = cfg.get("camera", {})
    min_box = ccfg.get("min_box_height", 0.15)
    colcfg = cfg.get("collision", {})
    checker = (CollisionChecker(arm, meshes, colcfg.get("depth_mm", 0.3) * 0.001, colcfg.get("max_new_points", 3))
               if colcfg.get("enabled", True) else None)
    if checker:
        print(f"[synth] めり込み検査: {len(checker.pairs)} 組（T ポーズで接している組 {len(checker.base)}）")
    written, rejected = 0, 0
    for idx in range(args.start, args.start + args.count):
        rng = random.Random(args.seed * 1_000_003 + idx)
        # めり込みの無いポーズが出るまで引き直す
        for _ in range(colcfg.get("max_tries", 30)):
            # 一定確率で gamma を下げ、可動域いっぱいの大きなポーズ（腕を真横・開脚など）も出す
            gamma = (cfg.get("pose_gamma_wide", 1.0) if rng.random() < cfg.get("pose_wide_prob", 0.0)
                     else cfg.get("pose_gamma", 1.0))
            angles = randomize_pose(arm, cfg["joint_limits_deg"], cfg.get("neutral_prob", 0.25), rng,
                                    gamma, cfg.get("pose_center_deg"), cfg.get("bend"), cfg["keypoints"], checker)
            hit = checker.collides() if checker else None
            if hit is None:
                break
            rejected += 1
        else:
            print(f"[synth] {idx}: めり込まないポーズが見つからないのでスキップ")
            continue
        if mats:
            randomize_materials(meshes, mats, mcfg, rng)
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        verts = figure_vertices(meshes, depsgraph)
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        center, size = Vector((lo + hi) / 2), float((hi - lo).max())

        # フィギュアがほぼ全部写り、小さすぎない構図になるまでカメラを引き直す
        for _ in range(50):
            cam_info = place_camera(cam, center, size, ccfg, rng)
            bpy.context.view_layer.update()
            uv = project(scene, cam, verts)
            front = uv[uv[:, 2] > 0]
            if len(front) < len(uv):
                continue
            x0, y0 = front[:, :2].min(axis=0)
            x1, y1 = front[:, :2].max(axis=0)
            full = (x1 - x0) * (y1 - y0)
            cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, 1), min(y1, 1)
            inside = max(cx1 - cx0, 0) * max(cy1 - cy0, 0)
            if full > 0 and inside / full >= 0.9 and (cy1 - cy0) >= min_box:
                break
        else:
            print(f"[synth] {idx}: 構図が決まらないのでスキップ")
            continue

        randomize_lights(lights, world_bg, center, lcfg, rng, hdri)
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()

        tol = cfg.get("occlusion_tolerance", 0.06) * size
        kp_world = [keypoint_world(arm, cfg["keypoints"][n]) for n in kp_names]
        kp_uv = project(scene, cam, np.array([list(p) for p in kp_world]))
        cam_loc = cam.matrix_world.translation.copy()
        kps = []
        for p, uvz in zip(kp_world, kp_uv):
            v = visibility(scene, depsgraph, cam_loc, p, uvz, tol)
            kps += [float(uvz[0]), float(uvz[1]), v] if v else [0.0, 0.0, 0]

        stem = f"{idx:06d}"
        box = [(cx0 + cx1) / 2, (cy0 + cy1) / 2, cx1 - cx0, cy1 - cy0]
        label = "0 " + " ".join(f"{b:.6f}" for b in box) + " " + " ".join(
            f"{k:.6f}" if i % 3 != 2 else str(k) for i, k in enumerate(kps))
        (out / "labels" / f"{stem}.txt").write_text(label + "\n", encoding="utf-8")
        meta = {
            "index": idx,
            "resolution": [scene.render.resolution_x, scene.render.resolution_y],
            "camera": cam_info,
            "figure_size": size,
            "joint_angles_deg": angles,
            "keypoints_cam": {n: camera_space(cam, p) for n, p in zip(kp_names, kp_world)},
        }
        (out / "meta" / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

        if not args.dry_run:
            scene.render.filepath = str((out / "images" / f"{stem}.png").resolve())
            bpy.ops.render.render(write_still=True)
        written += 1
        print(f"[synth] {stem} 完了 ({written}/{args.count})")

    print(f"[synth] {written} 枚を {out} に書き出しました（めり込みで引き直したポーズ {rejected} 回）")


if __name__ == "__main__":
    main()
