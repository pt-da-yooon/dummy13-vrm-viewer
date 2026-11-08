"""YOLO-pose ラベルを画像に描いて確認する（合成データ・実写データ共通）。

使い方:
    python synth/preview.py data/synth/yolo/images/train --n 16
    python synth/preview.py data/synth/raw/images
緑=見えている, 橙=隠れている（v=1）。出力は既定で data/preview/。
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from common import DEFAULT_KEYPOINTS, REPO, load_keypoint_config

COLORS = {2: (0, 220, 0), 1: (0, 160, 255)}


def draw(img: np.ndarray, label: str, kp: dict) -> np.ndarray:
    h, w = img.shape[:2]
    for line in label.strip().splitlines():
        vals = [float(v) for v in line.split()]
        cx, cy, bw, bh = vals[1:5]
        cv2.rectangle(img, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)),
                      (int((cx + bw / 2) * w), int((cy + bh / 2) * h)), (255, 128, 0), 1)
        pts = np.array(vals[5:]).reshape(-1, 3)
        for a, b in kp["skeleton"]:
            if pts[a, 2] > 0 and pts[b, 2] > 0:
                cv2.line(img, (int(pts[a, 0] * w), int(pts[a, 1] * h)),
                         (int(pts[b, 0] * w), int(pts[b, 1] * h)), (200, 200, 200), 1, cv2.LINE_AA)
        for i, (x, y, v) in enumerate(pts):
            if v > 0:
                c = (int(x * w), int(y * h))
                cv2.circle(img, c, 3, COLORS[int(v)], -1, cv2.LINE_AA)
                cv2.putText(img, kp["names"][i], (c[0] + 4, c[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            COLORS[int(v)], 1, cv2.LINE_AA)
    return img


def main() -> None:
    p = argparse.ArgumentParser(description="YOLO-pose ラベルの可視化")
    p.add_argument("images", help="画像フォルダ（ラベルは images→labels に置き換えたパスから探す）")
    p.add_argument("--out", default=str(REPO / "data" / "preview"))
    p.add_argument("--n", type=int, default=16, help="ランダムに選ぶ枚数（0 で全部）")
    p.add_argument("--keypoints", default=str(DEFAULT_KEYPOINTS))
    args = p.parse_args()

    kp = load_keypoint_config(args.keypoints)
    images = sorted(f for f in Path(args.images).iterdir() if f.suffix.lower() in {".jpg", ".png"})
    if args.n and len(images) > args.n:
        images = sorted(random.sample(images, args.n))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in images:
        label_path = Path(str(f.parent).replace("images", "labels")) / f"{f.stem}.txt"
        if not label_path.exists():
            continue
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 4:  # 透過画像は灰色背景に載せる
            a = img[:, :, 3:4] / 255.0
            img = (img[:, :, :3] * a + 128 * (1 - a)).astype(np.uint8)
        cv2.imwrite(str(out / f"{f.stem}.jpg"), draw(img, label_path.read_text(), kp))
    print(f"{len(images)} 枚を {out} に書き出しました")


if __name__ == "__main__":
    main()
