"""Dummy13 の STL（assets/dummy13/）からリグ入りの .blend を自動生成する。

配置は synth/dummy13_layout.py で計算し、各パーツを対応するボーンに親子付けする
（分割パーツを剛体としてボーンに付ける方式なのでウェイト付けは不要）。

使い方:
    python synth/build_rig.py                       # pip の bpy モジュール
    blender -b -P synth/build_rig.py -- --hands fist  # Blender 本体
出力: models/dummy13_rig.blend（アーマチュア名 Dummy13_Rig、単位 m、足裏が Z=0）
"""

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "synth"))
import dummy13_layout as layout  # noqa: E402

MM = 0.001  # STL は mm、Blender はメートルで扱う


def parse_args() -> argparse.Namespace:
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    elif "-P" in sys.argv or "--python" in sys.argv:  # blender -b -P script.py（スクリプト用の引数なし）
        argv = []
    else:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description="Dummy13 リグ生成")
    p.add_argument("--assets", default=str(REPO / "assets" / "dummy13"))
    p.add_argument("--out", default=str(REPO / "models" / "dummy13_rig.blend"))
    p.add_argument("--hands", default="open", choices=["open", "fist", "grip"])
    p.add_argument("--armature", default="Dummy13_Rig")
    p.add_argument("--no-armor", action="store_true", help="アーマーを付けずフレームだけにする")
    return p.parse_args(argv)


def import_stl(path: Path) -> bpy.types.Mesh:
    """STL を読み込んでメッシュデータだけ返す（mm → m に縮小済み）。"""
    before = set(bpy.data.objects)
    if hasattr(bpy.ops.wm, "stl_import"):  # Blender 4.1 以降
        bpy.ops.wm.stl_import(filepath=str(path))
    else:
        bpy.ops.import_mesh.stl(filepath=str(path))
    obj = next(o for o in bpy.data.objects if o not in before)
    mesh = obj.data
    mesh.transform(Matrix.Scale(MM, 4))
    mesh.name = path.stem
    bpy.data.objects.remove(obj)
    return mesh


def to_blender(M: np.ndarray, offset: np.ndarray) -> Matrix:
    M = M.copy()
    M[:3, 3] = (M[:3, 3] + offset) * MM
    return Matrix(M.tolist())


def main() -> None:
    args = parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    lay = layout.build(hands=args.hands, armor=not args.no_armor)
    assets = Path(args.assets)

    meshes = {}
    for p in lay.parts:
        if p.stl not in meshes:
            meshes[p.stl] = import_stl(assets / f"{p.stl}.stl")

    # 足裏が Z=0 に来るようにずらす
    zmin = min(
        (np.asarray([v.co for v in meshes[p.stl].vertices]) / MM @ p.matrix[:3, :3].T + p.matrix[:3, 3])[:, 2].min()
        for p in lay.parts)
    offset = np.array([0.0, 0.0, -zmin])

    scene = bpy.context.scene
    arm_data = bpy.data.armatures.new(args.armature)
    arm = bpy.data.objects.new(args.armature, arm_data)
    scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    front = Vector((0, -1, 0))
    for name, (head, tail, parent) in layout.BONES.items():
        eb = arm_data.edit_bones.new(name)
        eb.head = Vector((lay.joints[head] + offset) * MM)
        eb.tail = Vector((lay.joints[tail] + offset) * MM)
        # ボーンのローカル X 軸がヒンジ軸（左右 or 上下）になるよう、Z 軸を正面に向ける。
        # 前後を向いたボーン（足）は Z 軸を上に向ける。
        eb.align_roll(front if abs(eb.vector.normalized().dot(front)) < 0.9 else Vector((0, 0, 1)))
    for name, (_, _, parent) in layout.BONES.items():
        if parent:
            arm_data.edit_bones[name].parent = arm_data.edit_bones[parent]
    bpy.ops.object.mode_set(mode="OBJECT")

    materials = {}
    for key, color in (("Frame", (0.08, 0.08, 0.09, 1)), ("Armor", (0.8, 0.25, 0.05, 1))):
        mat = bpy.data.materials.new(f"Dummy13_{key}")
        if bpy.app.version < (5, 0, 0):
            mat.use_nodes = True
        bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.5
        materials[key] = mat

    coll = bpy.data.collections.new("Dummy13_Parts")
    scene.collection.children.link(coll)
    for p in lay.parts:
        obj = bpy.data.objects.new(p.name, meshes[p.stl])
        coll.objects.link(obj)
        # メッシュは同じ STL のパーツで共有しているので、材質はオブジェクト側に付ける
        if not obj.material_slots:
            obj.data.materials.append(None)
        obj.material_slots[0].link = "OBJECT"
        obj.material_slots[0].material = materials["Armor" if p.name.startswith("armor_") else "Frame"]
        world = to_blender(p.matrix, offset)
        obj.parent = arm
        obj.parent_type = "BONE"
        obj.parent_bone = p.bone
        bpy.context.view_layer.update()
        obj.matrix_world = world

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out.resolve()))
    print(f"[rig] パーツ {len(lay.parts)} 個 / ボーン {len(layout.BONES)} 本 → {out}")


if __name__ == "__main__":
    main()
