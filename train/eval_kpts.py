"""関節ごとのずれを合成データの検証用画像で数える（どこが・どんなポーズで・どれだけずれるか）。

mAP だけでは分からない「肘・膝が体の内側に縮む」などのくせを調べる。

使い方:
    python train/eval_kpts.py --weights runs/pose/dummy26b/weights/best.pt
    python train/eval_kpts.py --weights ... --n 2000        # 使う枚数（既定 1000）
ずれは「フィギュアの枠の長辺」に対する % で表す（14 cm のフィギュアなら 1% ≒ 1.4 mm）。
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent


def best(result):
    """一番自信のある 1 体の (box xyxy, keypoints xy)。見つからなければ None"""
    if result.boxes is None or len(result.boxes) == 0:
        return None
    i = int(result.boxes.conf.argmax())
    return result.boxes.xyxy[i].cpu().numpy(), result.keypoints.xy[i].cpu().numpy()


def bend_deg(k3, a, b, c):
    """3D の曲げ角（0 = まっすぐ）"""
    v1, v2 = k3[a] - k3[b], k3[c] - k3[b]
    cos = v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return 180 - np.degrees(np.arccos(np.clip(cos, -1, 1)))


def main() -> None:
    p = argparse.ArgumentParser(description="関節ごとのずれの評価")
    p.add_argument("--weights", required=True)
    p.add_argument("--yolo", default=str(REPO / "data" / "synth" / "yolo"))
    p.add_argument("--meta", default=str(REPO / "data" / "synth" / "raw" / "meta"))
    p.add_argument("--n", type=int, default=1000, help="使う検証用画像の枚数")
    p.add_argument("--conf", type=float, default=0.25)
    args = p.parse_args()

    names = yaml.safe_load((REPO / "configs" / "keypoints.yaml").read_text(encoding="utf-8"))["keypoints"]
    idx = {n: i for i, n in enumerate(names)}
    model = YOLO(args.weights)
    imgsz = int(model.model.args.get("imgsz", 640)) if hasattr(model.model, "args") else 640
    files = sorted((Path(args.yolo) / "images" / "val").glob("*.jpg"))
    files = random.Random(0).sample(files, min(args.n, len(files)))

    # 曲げの深さで分けて見る関節: 関節 → (曲げ角を測る 3 点)
    bend_of = {}
    for s in "lr":
        arm = (idx[f"{s}_shoulder"], idx[f"{s}_elbow"], idx[f"{s}_wrist"])
        leg = (idx[f"{s}_hip"], idx[f"{s}_knee"], idx[f"{s}_ankle"])
        bend_of.update({idx[f"{s}_elbow"]: arm, idx[f"{s}_wrist"]: arm, idx[f"{s}_knee"]: leg, idx[f"{s}_ankle"]: leg})

    rows = {"推定": []}   # (関節, ずれ %, 曲げ角 or None)
    shrink = {k: [] for k in rows}                     # 手足の「肩〜手首」などの長さの比（推定 / 正解）
    missed = 0
    for f in files:
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        v = np.array((Path(args.yolo) / "labels" / "val" / f"{f.stem}.txt").read_text().split()[1:], np.float32)
        gt = v[4:].reshape(-1, 3)
        gxy, vis = gt[:, :2] * [w, h], gt[:, 2] > 0
        scale = max(v[2] * w, v[3] * h)
        meta = json.loads((Path(args.meta) / f"{f.stem.rsplit('_', 1)[0]}.json").read_text(encoding="utf-8"))
        k3 = {i: np.array(meta["keypoints_cam"][n]) for n, i in idx.items()}
        r = best(model.predict(img, imgsz=imgsz, conf=args.conf, verbose=False)[0])
        if r is None:
            missed += 1
            continue
        preds = {"推定": r[1]}
        for key, pxy in preds.items():
            err = np.linalg.norm(pxy - gxy, axis=1) / scale * 100
            for j in np.nonzero(vis)[0]:
                rows[key].append((j, err[j], bend_deg(k3, *bend_of[j]) if j in bend_of else None))
            for a, c in ((idx["l_shoulder"], idx["l_wrist"]), (idx["r_shoulder"], idx["r_wrist"]),
                         (idx["l_hip"], idx["l_ankle"]), (idx["r_hip"], idx["r_ankle"])):
                if vis[a] and vis[c] and np.linalg.norm(gxy[c] - gxy[a]) > 0.15 * scale:
                    shrink[key].append(np.linalg.norm(pxy[c] - pxy[a]) / np.linalg.norm(gxy[c] - gxy[a]))

    print(f"{len(files)} 枚（見つからなかった {missed} 枚）。ずれは枠の長辺に対する %、PCK5 は 5% 以内に入った割合")
    for key, rs in rows.items():
        a = np.array([(j, e, -1 if b is None else b) for j, e, b in rs])
        print(f"\n== {key} ==  全体: 平均 {a[:, 1].mean():.2f}%  PCK5 {np.mean(a[:, 1] < 5) * 100:.1f}%")
        print(f"  {'関節':<12s} {'平均':>6s} {'PCK5':>6s}   深く曲げたとき（90° 以上）")
        for j, n in enumerate(names):
            e = a[a[:, 0] == j]
            line = f"  {n:<12s} {e[:, 1].mean():5.2f}% {np.mean(e[:, 1] < 5) * 100:5.1f}%"
            deep = e[e[:, 2] >= 90] if j in bend_of else e[:0]
            if len(deep):
                line += f"   {deep[:, 1].mean():5.2f}% {np.mean(deep[:, 1] < 5) * 100:5.1f}%  ({len(deep)} 件)"
            print(line)
        s = np.array(shrink[key])
        print(f"  手足の付け根〜先の長さ（推定 / 正解）：中央値 {np.median(s):.3f}（1 より小さいと内側に縮んでいる）")


if __name__ == "__main__":
    main()
