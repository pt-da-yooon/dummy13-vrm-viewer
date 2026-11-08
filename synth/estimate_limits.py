"""部品どうしの干渉から関節の可動域を自動で求める。

T ポーズから各ボーンを1軸ずつ少しずつ回し、動く側の部品が動かない側の部品に
めり込み始める角度を上限とする（T ポーズ時点で既に接している分は差し引く）。
回転軸は build_rig.py のボーンと同じ（ローカル X = 「+ で前に曲がる」、Y = ねじり、Z = 横）。

使い方:
    python synth/estimate_limits.py              # 結果を表示
    python synth/estimate_limits.py --write      # rig_config.json の joint_limits_deg を書き換える
必要なもの: numpy, scipy, trimesh（Blender は不要）
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "synth"))
import dummy13_layout as layout  # noqa: E402

PITCH = 0.25      # SDF の格子間隔（mm）
DEPTH = 0.3       # これより深くめり込んだ点を「干渉」と数える（mm）
NEW_HITS = 3      # T ポーズより干渉点がこれだけ増えたら止める
STEP = 2.5        # 角度の刻み（度）
MAX_DEG = 150.0   # ヒンジ（膝・肘）はこれ以上回さない
MAX_BALL = 90.0   # ボールジョイントのスイングの上限（ソケットの縁の当たりは判定から外しているので控えめに）
MARGIN = 2.5      # 求めた上限から少し内側に戻す（度）
TWIST_MAX = 90.0  # ねじり（Y 軸）は干渉で止まらないことが多いので上限を決めておく（度）
# 関節の中心からこの距離以内は関節機構（ボールとソケット、ピンとクリップ、クリック用の歯）なので
# 干渉判定から外す（mm）。ボールジョイントはソケットの縁がボールの首に当たる分まで外す。
JOINT_R_BALL = 6.0
JOINT_R_HINGE = 4.0
HINGE_BONES = ("elbow", "forearm", "knee", "shin", "foot")


class PartSDF:
    """パーツのローカル座標での符号付き距離場（内側が負）。"""

    def __init__(self, mesh: trimesh.Trimesh):
        lo = mesh.bounds[0] - 1.0
        hi = mesh.bounds[1] + 1.0
        shape = np.ceil((hi - lo) / PITCH).astype(int) + 1
        g = np.stack(np.meshgrid(*[lo[i] + np.arange(shape[i]) * PITCH for i in range(3)], indexing="ij"), -1)
        vox = mesh.voxelized(PITCH).fill()
        idx = vox.points_to_indices(g.reshape(-1, 3))
        ok = np.all((idx >= 0) & (idx < np.array(vox.shape)), axis=1)
        inside = np.zeros(len(idx), bool)
        inside[ok] = vox.matrix[tuple(idx[ok].T)]
        inside = inside.reshape(shape)
        self.d = np.where(inside, -ndimage.distance_transform_edt(inside),
                          ndimage.distance_transform_edt(~inside)).astype(np.float32) * PITCH
        self.lo, self.hi = lo, hi

    def depth(self, p: np.ndarray) -> np.ndarray:
        """点のめり込み深さ（外側は 0）。"""
        out = np.zeros(len(p), np.float32)
        m = np.all((p > self.lo) & (p < self.hi), axis=1)
        if m.any():
            c = (p[m] - self.lo) / PITCH
            out[m] = np.clip(-ndimage.map_coordinates(self.d, c.T, order=1, mode="nearest"), 0, None)
        return out


def bone_basis(head, tail) -> np.ndarray:
    """build_rig.py の align_roll と同じ向きのボーン座標軸（列が x, y, z）。"""
    y = np.asarray(tail, float) - head
    y /= np.linalg.norm(y)
    front = np.array([0.0, -1.0, 0.0]) if abs(y @ [0, -1, 0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    z = front - (front @ y) * y
    z /= np.linalg.norm(z)
    return np.column_stack([np.cross(y, z), y, z])


def axis_rot(axis: int, deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    R = np.eye(3)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def main() -> None:
    p = argparse.ArgumentParser(description="干渉から可動域を推定")
    p.add_argument("--config", default=str(REPO / "synth" / "rig_config.json"))
    p.add_argument("--write", action="store_true", help="rig_config.json を書き換える")
    p.add_argument("--bones", nargs="*", help="対象ボーン（省略時は rig_config.json の全ボーンの左側）")
    args = p.parse_args()

    lay = layout.build()
    J = lay.joints
    assets = REPO / "assets" / "dummy13"
    meshes, sdfs, samples = {}, {}, {}
    for part in lay.parts:
        if part.stl not in meshes:
            m = trimesh.load(assets / f"{part.stl}.stl")
            meshes[part.stl] = m
            sdfs[part.stl] = PartSDF(m)
            samples[part.stl] = trimesh.sample.sample_surface_even(m, 400, seed=0)[0]
    print(f"[limits] SDF 準備完了（{len(meshes)} 種類）", flush=True)

    children = {}
    for b, (_, _, parent) in layout.BONES.items():
        children.setdefault(parent, []).append(b)

    def subtree(b):
        out = [b]
        for c in children.get(b, []):
            out += subtree(c)
        return out

    def world_points(part):
        s = samples[part.stl]
        return s @ part.matrix[:3, :3].T + part.matrix[:3, 3]

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    limits = cfg["joint_limits_deg"]
    bones = args.bones or [b for b in limits if not b.endswith(".R")]

    result = {}
    for bone in bones:
        head_j, tail_j, _ = layout.BONES[bone]
        h = J[head_j]
        B = bone_basis(h, J[tail_j])
        moving_bones = set(subtree(bone))
        moving = [pt for pt in lay.parts if pt.bone in moving_bones]
        static = [pt for pt in lay.parts if pt.bone not in moving_bones]
        P0 = np.concatenate([world_points(pt) for pt in moving])
        jr = JOINT_R_HINGE if bone.split(".")[0] in HINGE_BONES else JOINT_R_BALL
        P0 = P0[np.linalg.norm(P0 - h, axis=1) > jr]
        inv = [np.linalg.inv(pt.matrix) for pt in static]

        def hits(R):
            P = (P0 - h) @ (B @ R @ B.T).T + h
            counts = []
            for pt, Mi in zip(static, inv):
                q = P @ Mi[:3, :3].T + Mi[:3, 3]
                counts.append(int((sdfs[pt.stl].depth(q) > DEPTH).sum()))
            return np.array(counts)

        base = hits(np.eye(3))
        result[bone] = {}
        for axis, name in enumerate("xyz"):
            if name not in limits[bone]:
                continue
            rng = []
            max_deg = TWIST_MAX if name == "y" else (MAX_DEG if jr == JOINT_R_HINGE else MAX_BALL)
            for sign in (-1, 1):
                ang, hit_by = 0.0, None
                while ang + STEP <= max_deg:
                    c = hits(axis_rot(axis, sign * (ang + STEP)))
                    bad = np.nonzero(c - base >= NEW_HITS)[0]
                    if len(bad):
                        hit_by = static[bad[np.argmax((c - base)[bad])]].name
                        break
                    ang += STEP
                lim = max(0.0, ang - MARGIN) if hit_by else ang
                rng.append((sign * lim, hit_by))
            result[bone][name] = [round(rng[0][0], 1), round(rng[1][0], 1)]
            print(f"{bone:12s} {name}: {rng[0][0]:7.1f} ({rng[0][1] or '制限なし'}) 〜 "
                  f"{rng[1][0]:6.1f} ({rng[1][1] or '制限なし'})", flush=True)

    if args.write:
        for bone, axes in result.items():
            limits[bone].update(axes)
            rb = bone[:-2] + ".R" if bone.endswith(".L") else None
            if rb and rb in limits:  # 右は Y・Z の符号が逆
                limits[rb].update({a: ([-v[1], -v[0]] if a in "yz" else v) for a, v in axes.items()})
        text = json.dumps(cfg, ensure_ascii=False, indent=2)
        Path(args.config).write_text(text + "\n", encoding="utf-8")
        print(f"[limits] {args.config} を更新しました")


if __name__ == "__main__":
    main()
