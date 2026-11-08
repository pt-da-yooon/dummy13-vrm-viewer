"""YOLO-pose で Dummy13 のキーポイント検出モデルを学習する。

前段: synth/generate.py → synth/composite.py で data/synth/yolo/dataset.yaml ができている前提。

使い方:
    python train/train.py                               # 既定（yolo26n-pose から微調整、100 エポック）
    python train/train.py --model yolo26s-pose.pt --epochs 200 --batch 32
    python train/train.py --model yolo26n-pose.yaml     # 事前学習なしで一から
    python train/train.py --resume                      # 途中で止めた学習を続きから（weights/last.pt から再開）
    python train/train.py --model runs/pose/dummy13c/weights/best.pt --name dummy13r --epochs 10 --real data/real
                                                        # 実物の写真（web/label.html で点付け）を混ぜて追加学習
結果は runs/pose/<name>/（weights/best.pt が学習済みの重み）。
"""

import argparse
import random
import zlib
from pathlib import Path

import yaml
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description="Dummy13 YOLO-pose 学習")
    p.add_argument("--data", default=str(REPO / "data" / "synth" / "yolo" / "dataset.yaml"))
    p.add_argument("--model", default="yolo26n-pose.pt", help="事前学習済みの重み（.pt）か構成（.yaml）")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="例: 0（GPU）/ cpu。省略時は自動")
    p.add_argument("--name", default="dummy13")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", action="store_true",
                   help="runs/pose/<name>/weights/last.pt から続きを学習（エポック数などの設定は前回のまま）")
    p.add_argument("--real", help="実物の写真のフォルダ（images/ と labels/。web/label.html で点付けしたもの）")
    p.add_argument("--real-repeat", type=int, default=10, help="実物の写真を学習で何回繰り返すか（合成より少ないので水増し）")
    p.add_argument("--real-val", type=float, default=0.2, help="実物の写真のうち評価に回す割合（評価は実物だけで行う）")
    p.add_argument("--synth-fraction", type=float, default=1.0, help="合成データのうち使う割合（追加学習を短くしたいとき）")
    args = p.parse_args()
    if args.real:
        args.data = make_mixed_data(args)

    if args.resume:
        last = REPO / "runs" / "pose" / args.name / "weights" / "last.pt"
        if not last.exists():
            raise SystemExit(f"{last} がありません（まだ 1 エポックも終わっていない？）")
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        train(model, args)
    metrics = model.val(project=str(REPO / "runs" / "pose"), name=f"{args.name}_val", exist_ok=True)
    print(f"pose mAP50-95: {metrics.pose.map:.4f}  mAP50: {metrics.pose.map50:.4f}")


def make_mixed_data(args) -> str:
    """合成データ＋実物の写真の学習リストを作る。評価は実物の写真だけで行う（実物での精度が分かる）。"""
    synth = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    root = Path(synth["path"])
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    synth_train = sorted(f for f in (root / synth["train"]).glob("*") if f.suffix.lower() in exts)
    if args.synth_fraction < 1.0:
        synth_train = random.Random(0).sample(synth_train, int(len(synth_train) * args.synth_fraction))
    real = Path(args.real).resolve()
    # ラベルの無い写真は「Dummy13 が写っていない」扱いになってしまうので、点付けしたものだけ使う
    real_imgs = sorted(f for f in (real / "images").glob("*")
                       if f.suffix.lower() in exts and (real / "labels" / f"{f.stem}.txt").exists())
    if not real_imgs:
        raise SystemExit(f"{real}/labels にラベルがありません（web/label.html で点付けしてください）")
    # 写真の名前で決まる分け方にして、写真を足しても評価用が入れ替わらないようにする
    is_val = lambda f: zlib.crc32(f.name.encode()) % 1000 < args.real_val * 1000
    real_val = [f for f in real_imgs if is_val(f)]
    real_train = [f for f in real_imgs if not is_val(f)]
    out = REPO / "runs" / "pose" / f"{args.name}_data"
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.txt").write_text("\n".join(str(f) for f in synth_train + real_train * args.real_repeat) + "\n", encoding="utf-8")
    if real_val:
        (out / "val.txt").write_text("\n".join(str(f) for f in real_val) + "\n", encoding="utf-8")
        val = str(out / "val.txt")
    else:
        val = str(root / synth["val"])
        print("※ 実物の写真が少ないので、評価は合成データで行います")
    data = {**synth, "path": str(out), "train": str(out / "train.txt"), "val": val}
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"学習: 合成 {len(synth_train)} 枚 + 実物 {len(real_train)} 枚 × {args.real_repeat} ／ 評価: "
          f"{'実物 ' + str(len(real_val)) + ' 枚' if real_val else '合成'}")
    return str(out / "data.yaml")


def train(model, args) -> None:
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(REPO / "runs" / "pose"),
        name=args.name,
        exist_ok=True,
        # 合成データ側で色・背景・照明は十分ばらつかせているので、色の拡張は控えめに。
        # 左右反転は dataset.yaml の flip_idx で左右のキーポイントを入れ替えてくれる。
        hsv_h=0.01, hsv_s=0.4, hsv_v=0.3,
        degrees=30.0, fliplr=0.5, mosaic=0.5,
    )


if __name__ == "__main__":
    main()
