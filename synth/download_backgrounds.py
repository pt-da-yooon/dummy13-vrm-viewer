"""背景用の素材を Poly Haven（CC0）からダウンロードし、背景画像を作る。

- テクスチャ（木・布・床・壁など）の色画像 → data/backgrounds/textures/
- 屋内の HDRI（360° 画像）→ data/hdri/（generate.py の照明にも使える）
  さらに HDRI からランダムな向きの透視投影画像を切り出す → data/backgrounds/hdri_views/

使い方:
    python synth/download_backgrounds.py                    # 既定（テクスチャ全部 + 屋内 HDRI 全部）
    python synth/download_backgrounds.py --max-hdris 50     # HDRI の数を絞る
途中で止めても、ダウンロード済みのファイルは飛ばして再開できる。

Poly Haven の素材は CC0（https://polyhaven.com/license）。
"""

import argparse
import hashlib
import json
import math
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
API = "https://api.polyhaven.com"
HEADERS = {"User-Agent": "dummy13-vrm-viewer-synth/0.1 (+https://github.com/pt-da-yooon/dummy13-vrm-viewer)"}  # 既定の UA は弾かれる
TEXTURE_CATEGORIES = {"wood", "fabric", "floor", "wall", "plaster", "plaster-concrete", "indoor", "metal",
                      "tiles", "clean", "man made", "concrete", "brick"}


def get_json(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
        return json.load(r)


def download(url: str, dst: Path, md5: str | None = None) -> Path:
    if dst.exists() and (md5 is None or hashlib.md5(dst.read_bytes()).hexdigest() == md5):
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dst)
    return dst


def equirect_view(pano: np.ndarray, yaw: float, pitch: float, fov: float, w: int, h: int) -> np.ndarray:
    """360° 画像（正距円筒図法）から透視投影の画像を切り出す。角度はラジアン。"""
    f = (w / 2) / math.tan(fov / 2)
    x, y = np.meshgrid(np.arange(w) - w / 2 + 0.5, np.arange(h) - h / 2 + 0.5)
    d = np.stack([x, -y, np.full_like(x, f)], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    cp, sp, cy, sy = math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    d = d @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]).T  # 上下
    d = d @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]).T  # 左右
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arcsin(np.clip(d[..., 1], -1, 1))
    ph, pw = pano.shape[:2]
    mx = ((lon / (2 * math.pi) + 0.5) * pw).astype(np.float32)
    my = ((0.5 - lat / math.pi) * ph).astype(np.float32)
    return cv2.remap(pano, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def tonemap(hdr: np.ndarray, rng: random.Random) -> np.ndarray:
    """HDR を 8bit に（露出をランダムに変えて Reinhard + ガンマ）。"""
    lum = 0.2126 * hdr[..., 2] + 0.7152 * hdr[..., 1] + 0.0722 * hdr[..., 0]
    key = rng.uniform(0.12, 0.3) / (np.exp(np.mean(np.log(lum + 1e-6))) + 1e-6)
    x = hdr * key
    x = x / (1 + x)
    return (np.clip(x, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description="Poly Haven から背景素材を取得")
    p.add_argument("--out", default=str(REPO / "data"))
    p.add_argument("--max-textures", type=int, default=0, help="0 で全部")
    p.add_argument("--max-hdris", type=int, default=0, help="0 で全部")
    p.add_argument("--hdri-res", default="2k", choices=["1k", "2k", "4k"])
    p.add_argument("--views-per-hdri", type=int, default=12)
    p.add_argument("--view-size", type=int, nargs=2, default=[960, 720])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out)
    rng = random.Random(args.seed)

    # --- テクスチャ
    tex = get_json(f"{API}/assets?t=textures")
    tex_ids = sorted(k for k, v in tex.items() if TEXTURE_CATEGORIES & set(v.get("categories", [])))
    if args.max_textures:
        tex_ids = tex_ids[:args.max_textures]
    print(f"テクスチャ {len(tex_ids)} 件", flush=True)

    def fetch_texture(aid):
        try:
            files = get_json(f"{API}/files/{aid}")
            jpg = files["Diffuse"]["1k"]["jpg"]
            download(jpg["url"], out / "backgrounds" / "textures" / f"{aid}.jpg", jpg["md5"])
        except Exception as e:  # noqa: BLE001（1件の失敗で全体を止めない）
            print(f"  失敗 {aid}: {e}", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(fetch_texture, tex_ids))

    # --- 屋内 HDRI
    hdris = get_json(f"{API}/assets?t=hdris")
    hdri_ids = sorted(k for k, v in hdris.items() if "indoor" in v.get("categories", []))
    if args.max_hdris:
        hdri_ids = hdri_ids[:args.max_hdris]
    print(f"屋内 HDRI {len(hdri_ids)} 件", flush=True)

    done = [0]

    def fetch_hdri(aid):
        dst = out / "hdri" / f"{aid}_{args.hdri_res}.hdr"
        try:
            if dst.exists():  # 取得済み（md5 の確認は省略して速く再開する）
                return dst
            f = get_json(f"{API}/files/{aid}")["hdri"][args.hdri_res]["hdr"]
            return download(f["url"], dst, f["md5"])
        except Exception as e:  # noqa: BLE001
            print(f"  失敗 {aid}: {e}", flush=True)
        finally:
            done[0] += 1
            if done[0] % 10 == 0 or done[0] == len(hdri_ids):
                print(f"  HDRI {done[0]}/{len(hdri_ids)}", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        paths = [p for p in ex.map(fetch_hdri, hdri_ids) if p]

    # --- HDRI から背景画像を切り出す（机の上のフィギュアを撮る想定で、やや見下ろし気味）
    vdir = out / "backgrounds" / "hdri_views"
    vdir.mkdir(parents=True, exist_ok=True)
    w, h = args.view_size
    print("HDRI から背景を切り出し中…", flush=True)
    for path in paths:
        if (vdir / f"{path.stem}_{args.views_per_hdri - 1:02d}.jpg").exists():
            continue
        pano = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if pano is None:
            continue
        for i in range(args.views_per_hdri):
            view = equirect_view(pano, rng.uniform(-math.pi, math.pi), math.radians(rng.uniform(-35, 15)),
                                 math.radians(rng.uniform(45, 80)), w, h)
            cv2.imwrite(str(vdir / f"{path.stem}_{i:02d}.jpg"), tonemap(view, rng), [cv2.IMWRITE_JPEG_QUALITY, 92])
    n_tex = len(list((out / "backgrounds" / "textures").glob("*.jpg")))
    n_view = len(list(vdir.glob("*.jpg")))
    print(f"完了: テクスチャ {n_tex} 枚, HDRI {len(paths)} 件, HDRI から切り出した背景 {n_view} 枚")


if __name__ == "__main__":
    main()
