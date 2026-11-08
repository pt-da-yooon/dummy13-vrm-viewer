"""generate.py の透過レンダリングに背景を合成し、YOLO-pose のデータセットにする。

背景画像フォルダを渡すとその中からランダムに選ぶ（部屋・机の写真などを集めておくとよい）。
無ければ単色・グラデーション・ノイズの背景を自動生成する。
Webカメラらしさを出すため、明るさ・ぼかし・ノイズ・JPEG 圧縮もランダムにかける。
手で持って撮ることが多いので、一部の画像には肌色の指・手のひらを体の上に重ねる（--occlude-prob）。

使い方:
    python synth/composite.py --backgrounds data/backgrounds --per-image 3

出力（既定 data/synth/yolo/）:
    images/{train,val}/*.jpg, labels/{train,val}/*.txt, dataset.yaml
"""

import argparse
import random
import shutil
import zlib
from pathlib import Path

import cv2
import numpy as np
import yaml

from common import DEFAULT_KEYPOINTS, REPO, load_keypoint_config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def random_background(w: int, h: int, bg_files: list[Path], rng: random.Random) -> np.ndarray:
    if bg_files and rng.random() < 0.9:
        img = cv2.imread(str(rng.choice(bg_files)), cv2.IMREAD_COLOR)
        if img is not None:
            # ランダムな位置・大きさで切り出して出力サイズに合わせる
            ih, iw = img.shape[:2]
            scale = rng.uniform(0.4, 1.0)
            cw = min(iw, int(ih * scale * w / h)) if iw * h > ih * w else int(iw * scale)
            ch = int(cw * h / w)
            ch, cw = min(ch, ih), min(cw, iw)
            x, y = rng.randint(0, iw - cw), rng.randint(0, ih - ch)
            img = cv2.resize(img[y:y + ch, x:x + cw], (w, h), interpolation=cv2.INTER_AREA)
            return cv2.flip(img, 1) if rng.random() < 0.5 else img
    kind = rng.choice(["solid", "gradient", "noise"])
    c1 = np.array([rng.randint(0, 255) for _ in range(3)], np.float32)
    if kind == "solid":
        return np.full((h, w, 3), c1, np.uint8)
    if kind == "gradient":
        c2 = np.array([rng.randint(0, 255) for _ in range(3)], np.float32)
        t = np.linspace(0, 1, h if rng.random() < 0.5 else w, dtype=np.float32)
        t = t[:, None, None] if len(t) == h else t[None, :, None]
        return np.broadcast_to(c1 * (1 - t) + c2 * t, (h, w, 3)).astype(np.uint8)
    # 低解像度ノイズを拡大したまだら模様
    small = np.random.default_rng(rng.randint(0, 2**31)).integers(
        0, 256, (rng.randint(2, 16), rng.randint(2, 16), 3), dtype=np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def composite(rgba: np.ndarray, bg: np.ndarray) -> np.ndarray:
    fg = rgba[:, :, :3].astype(np.float32)
    a = rgba[:, :, 3:4].astype(np.float32) / 255.0
    return (fg * a + bg.astype(np.float32) * (1 - a)).clip(0, 255).astype(np.uint8)


def add_hand_occluder(img: np.ndarray, alpha: np.ndarray, rng: random.Random) -> np.ndarray:
    """フィギュアを手で持っている状態を真似て、肌色の指（丸い棒）と手のひらを体の上に重ねる。

    キーポイントのラベルはそのまま（隠れた関節も位置を当てられるように学習させる）。
    """
    ys, xs = np.nonzero(alpha > 127)
    if len(xs) == 0:
        return img
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    fig_h = max(y1 - y0, x1 - x0, 20)
    h, w = img.shape[:2]
    if rng.random() < 0.8:  # 肌色（HSV、OpenCV の H は 0〜180）
        hsv = np.uint8([[[rng.randint(4, 18), rng.randint(50, 150), rng.randint(110, 245)]]])
    else:  # 袖・手袋・ほかの物
        hsv = np.uint8([[[rng.randint(0, 179), rng.randint(0, 255), rng.randint(30, 230)]]])
    color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)

    # 指先が体のどこかに来るようにし、指はそこから画面の外側へ向かって伸ばす
    i = rng.randrange(len(xs))
    tip = np.array([xs[i], ys[i]], np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    d = np.array([np.cos(ang), np.sin(ang)], np.float32)   # 指先 → 根元の向き
    n = np.array([-d[1], d[0]], np.float32)
    fw = fig_h * rng.uniform(0.07, 0.14)                    # 指の太さ
    nf = rng.choice([1, 1, 2, 2, 3, 4])
    lengths = []
    mask = np.zeros((h, w), np.uint8)
    shade = np.zeros((h, w), np.float32)
    for k in range(nf):
        off = (k - (nf - 1) / 2) * fw * rng.uniform(0.95, 1.1)
        length = fig_h * rng.uniform(0.25, 0.6)
        lengths.append(length)
        a = tip + n * off + d * rng.uniform(-0.2, 0.2) * fw
        b = a + d * length
        m = np.zeros((h, w), np.uint8)
        cv2.line(m, tuple(int(v) for v in a), tuple(int(v) for v in b), 255, max(int(fw), 2), cv2.LINE_AA)
        # 指ごとに縁を少し暗くして、隣の指との境目を出す
        edge = cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(fw * 0.25, 1.0)) / 255.0
        shade = np.maximum(shade, edge)
        mask = np.maximum(mask, m)
    if rng.random() < 0.5:  # 手のひら（指の根元側）
        half = fig_h * rng.uniform(0.15, 0.3)              # 指の向きの半径
        c = tip + d * (min(lengths) + half * 0.8)
        ax = (int(half), int(fw * max(nf, 2) * rng.uniform(0.55, 0.8)) + 2)
        cv2.ellipse(mask, tuple(int(v) for v in c), ax, float(np.degrees(ang)), 0, 360, 255, -1, cv2.LINE_AA)
        shade = np.maximum(shade, cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), fw * 0.4))
    # 光の当たり方（一方向のグラデーション）と境目の陰
    g = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    light = 1 + rng.uniform(0.05, 0.3) * ((xx - tip[0]) * np.cos(g) + (yy - tip[1]) * np.sin(g)) / fig_h
    tone = (light * (0.55 + 0.45 * shade))[..., None] * color
    a = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), rng.uniform(0.5, 1.5))[..., None]
    out = img.astype(np.float32) * (1 - a) + tone.clip(0, 255) * a
    return out.astype(np.uint8)


def augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    img = img.astype(np.float32)
    img = img * rng.uniform(0.7, 1.3) + rng.uniform(-30, 30)  # コントラスト・明るさ
    img = img * np.array([rng.uniform(0.9, 1.1) for _ in range(3)], np.float32)  # ホワイトバランス
    img = img.clip(0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.3:
        noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(2, 10), img.shape)
        img = (img + noise).clip(0, 255).astype(np.uint8)
    return img


def main() -> None:
    p = argparse.ArgumentParser(description="背景合成して YOLO-pose データセットを作る")
    p.add_argument("--raw", default=str(REPO / "data" / "synth" / "raw"))
    p.add_argument("--out", default=str(REPO / "data" / "synth" / "yolo"))
    p.add_argument("--backgrounds", default=str(REPO / "data" / "backgrounds"),
                   help="背景画像フォルダ（サブフォルダも探す。無ければ自動生成の背景のみ）")
    p.add_argument("--keypoints", default=str(DEFAULT_KEYPOINTS))
    p.add_argument("--per-image", type=int, default=1, help="1枚のレンダリングから作る合成画像の数")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--occlude-prob", type=float, default=0.35, help="手（指）で持っている状態を重ねる割合")
    args = p.parse_args()

    rng = random.Random(args.seed)
    raw, out = Path(args.raw), Path(args.out)
    bg_files = []
    if args.backgrounds and Path(args.backgrounds).is_dir():
        bg_files = sorted(f for f in Path(args.backgrounds).rglob("*") if f.suffix.lower() in IMAGE_EXTS)
        print(f"背景画像: {len(bg_files)} 枚")

    stems = sorted(f.stem for f in (raw / "images").glob("*.png") if (raw / "labels" / f"{f.stem}.txt").exists())
    if not stems:
        raise SystemExit(f"{raw} にレンダリング結果がありません（synth/generate.py を先に実行）")
    # 同じレンダリング由来の画像が train と val にまたがらないよう、元画像単位で分ける。
    # 分け方は名前のハッシュで決める（枚数を足して作り直しても、前に val だった画像は val のまま）
    splits = {"val": [], "train": []}
    for stem in stems:
        splits["val" if zlib.crc32(stem.encode()) % 10000 < args.val_ratio * 10000 else "train"].append(stem)

    # 前回の出力を消す（残すと、前回と今回で train / val が入れ替わった画像が両方に入り、評価が甘くなる）
    for d in ("images", "labels"):
        if (out / d).exists():
            shutil.rmtree(out / d)
    for split, names in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for stem in names:
            rgba = cv2.imread(str(raw / "images" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
                print(f"スキップ（RGBA ではない）: {stem}")
                continue
            h, w = rgba.shape[:2]
            for k in range(args.per_image):
                img = composite(rgba, random_background(w, h, bg_files, rng))
                if rng.random() < args.occlude_prob:
                    img = add_hand_occluder(img, rgba[:, :, 3], rng)
                img = augment(img, rng)
                name = f"{stem}_{k}"
                quality = rng.randint(40, 95)  # Webカメラの圧縮ノイズを模擬
                cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
                shutil.copyfile(raw / "labels" / f"{stem}.txt", out / "labels" / split / f"{name}.txt")
        print(f"{split}: {len(names)} 枚 × {args.per_image}")

    kp = load_keypoint_config(args.keypoints)
    dataset = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": [len(kp["names"]), 3],
        "flip_idx": kp["flip_idx"],
        "names": {0: "dummy13"},
    }
    (out / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"{out / 'dataset.yaml'} を書き出しました")


if __name__ == "__main__":
    main()
