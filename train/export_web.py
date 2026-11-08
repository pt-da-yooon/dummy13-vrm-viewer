"""学習済みの重みを、ブラウザ版（web/）で使える ONNX に書き出す。

使い方:
    python train/export_web.py                                  # runs/pose/dummy13/weights/best.pt を使う
    python train/export_web.py --weights runs/pose/quick/weights/best.pt
出力: web/models/dummy13-pose.onnx と web/models/model.json（キーポイント名・骨格・入力サイズ）
"""

import argparse
import json
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent


def clean_metadata(path: Path) -> None:
    """Ultralytics が ONNX に書き込む説明（description）には学習データのフルパスが入るので、公開しても困らない文に置き換える。"""
    import onnx as onnx_lib

    m = onnx_lib.load(str(path))
    for prop in m.metadata_props:
        if prop.key == "description":
            prop.value = "Ultralytics YOLO pose model for Dummy 13 (Dummy13 VRM Viewer)"
    onnx_lib.save(m, str(path))


def main() -> None:
    p = argparse.ArgumentParser(description="ブラウザ用 ONNX の書き出し")
    p.add_argument("--weights", default=str(REPO / "runs" / "pose" / "dummy13" / "weights" / "best.pt"))
    p.add_argument("--imgsz", type=int, help="入力サイズ（省略時は学習したときのサイズ。小さくすると速いが精度は下がる）")
    args = p.parse_args()

    out_dir = REPO / "web" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    imgsz = args.imgsz or int(model.model.args.get("imgsz", 640))
    onnx = Path(model.export(format="onnx", imgsz=imgsz, opset=17, simplify=True, dynamic=False))
    shutil.copyfile(onnx, out_dir / "dummy13-pose.onnx")
    clean_metadata(out_dir / "dummy13-pose.onnx")

    kcfg = yaml.safe_load((REPO / "configs" / "keypoints.yaml").read_text(encoding="utf-8"))
    meta = {"imgsz": imgsz, "keypoints": kcfg["keypoints"], "skeleton": kcfg["skeleton"]}
    (out_dir / "model.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    size = (out_dir / "dummy13-pose.onnx").stat().st_size / 1e6
    print(f"書き出しました: {out_dir / 'dummy13-pose.onnx'}（{size:.1f} MB, 入力 {imgsz}px）, model.json")


if __name__ == "__main__":
    main()
