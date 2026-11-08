"""画像から各関節の奥行き（手前／奥）を推定するモデルを学習する。

2D キーポイントの形だけでは「前に倒す」と「後ろに反らす」が区別できないことがある
（カメラがフィギュアと同じ高さだと 2D ではほぼ同じ形に写る）。そこで、フィギュアの部分を切り出した画像
（アーマーの見え方・陰影）と 2D キーポイントの両方から、腰を基準にした各関節の奥行きを当てる。

前段: synth/generate.py（レンダリングあり）→ synth/composite.py で
data/synth/yolo/（背景合成済みの画像とラベル）と data/synth/raw/meta/（3D の正解）ができている前提。

使い方:
    python train/train_depth.py --device 0                 # GPU で学習
    python train/train_depth.py --epochs 20 --batch 64
出力: web/models/depth.onnx と web/models/depth.json（ブラウザ版が読み込む）
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parent.parent

# web/avatar.js の BONE_MM と同じ（Dummy13 の骨の長さ、mm）
BONE_MM = {
    ("hips", "chest"): 34.0, ("chest", "neck"): 10.0, ("neck", "head"): 14.5,
    ("neck", "l_shoulder"): 14.32, ("l_shoulder", "l_elbow"): 22.2, ("l_elbow", "l_wrist"): 18.0,
    ("neck", "r_shoulder"): 14.32, ("r_shoulder", "r_elbow"): 22.2, ("r_elbow", "r_wrist"): 18.0,
    ("hips", "l_hip"): 8.0, ("l_hip", "l_knee"): 30.2, ("l_knee", "l_ankle"): 31.0,
    ("hips", "r_hip"): 8.0, ("r_hip", "r_knee"): 30.2, ("r_knee", "r_ankle"): 31.0,
}
UNIT = 100.0       # 奥行きは mm / 100 で扱う
CROP_SCALE = 1.25  # 検出枠の長辺 × これ の正方形で切り出す（web/app.js と同じ）


def crop_square(img: np.ndarray, box, size: int) -> np.ndarray:
    """枠の中心から正方形に切り出して size に縮小する。はみ出した所は灰色 114（web/app.js の cropForDepth と同じ）。"""
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) * CROP_SCALE
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sx, sy = cx - side / 2, cy - side / 2
    m = np.float32([[size / side, 0, -sx * size / side], [0, size / side, -sy * size / side]])
    return cv2.warpAffine(img, m, (size, size), flags=cv2.INTER_AREA, borderValue=(114, 114, 114))


def kp_input(uv: np.ndarray, mask: np.ndarray, bones, hips: int) -> np.ndarray:
    """2D（px, y 上向き）→ キーポイント側の入力。train_lifter.py の make_inputs と同じ計算。"""
    s = 0.0
    for a, b, L in bones:
        if mask[a] and mask[b]:
            s = max(s, float(np.linalg.norm(uv[b] - uv[a])) / L)
    x = np.zeros((len(uv), 2), np.float32)
    if s > 0:
        x = (uv - uv[hips]) / s / UNIT * mask[:, None]
    return np.concatenate([x.reshape(-1), mask]).astype(np.float32)


class DepthData(Dataset):
    def __init__(self, yolo: Path, meta: Path, split: str, names, flip, bones, size: int, train: bool):
        self.files = sorted((yolo / "images" / split).glob("*.jpg"))
        self.labels = yolo / "labels" / split
        self.names, self.flip, self.bones, self.size, self.train = names, flip, bones, size, train
        self.hips = names.index("hips")
        self.dz, self.elev = {}, {}
        for f in self.files:  # 3D の正解は元のレンダリング単位（000123_0.jpg → meta/000123.json）
            stem = f.stem.rsplit("_", 1)[0]
            if stem in self.dz:
                continue
            m = json.loads((meta / f"{stem}.json").read_text(encoding="utf-8"))
            z = np.array([m["keypoints_cam"][n][2] for n in names], np.float32)
            self.dz[stem] = -(z - z[self.hips]) * 1000 / UNIT  # 手前が +
            self.elev[stem] = m["camera"]["elevation_deg"]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        v = np.array(self.labels.joinpath(f.stem + ".txt").read_text().split()[1:], np.float32)
        cx, cy, bw, bh = v[:4] * [w, h, w, h]
        k = v[4:].reshape(-1, 3)
        uv = k[:, :2] * [w, h]
        mask = (k[:, 2] > 0).astype(np.float32)
        dz = self.dz[f.stem.rsplit("_", 1)[0]].copy()
        if self.train:  # 検出枠のずれ・キーポイントの欠けとずれ
            cx += random.uniform(-0.05, 0.05) * bw
            cy += random.uniform(-0.05, 0.05) * bh
            bw *= random.uniform(0.9, 1.1)
            bh *= random.uniform(0.9, 1.1)
            mask *= (np.random.rand(len(mask)) > 0.05)
            mask[self.hips] = (k[self.hips, 2] > 0)
            uv = uv + np.random.randn(*uv.shape) * max(bw, bh) * 0.01
        box = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
        crop = crop_square(img, box, self.size)
        uv = np.stack([uv[:, 0], -uv[:, 1]], 1)  # y 上向き
        if self.train and random.random() < 0.5:  # 左右反転（左右の関節を入れ替える）
            crop = crop[:, ::-1]
            uv = uv[self.flip] * [-1, 1]
            mask, dz = mask[self.flip], dz[self.flip]
        if self.train:
            crop = np.clip(crop.astype(np.float32) * random.uniform(0.8, 1.2) + random.uniform(-20, 20), 0, 255)
        x_img = torch.from_numpy(np.ascontiguousarray(crop, np.float32).transpose(2, 0, 1) / 255.0)
        x_kp = torch.from_numpy(kp_input(uv.astype(np.float32), mask, self.bones, self.hips))
        return x_img, x_kp, torch.from_numpy(dz)


class DepthNet(nn.Module):
    """画像（MobileNetV3）＋ 2D キーポイント → 各関節の奥行き（mm/100）。"""

    def __init__(self, k: int, pretrained: bool = True):
        super().__init__()
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
        net = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
        self.features = net.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.kp = nn.Sequential(nn.Linear(3 * k, 256), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(960 + 256, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, k))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image, kp):
        f = self.pool(self.features((image - self.mean) / self.std)).flatten(1)
        return self.head(torch.cat([f, self.kp(kp)], 1))


def evaluate(model, loader, data: DepthData, names, bones, device) -> None:
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for img, kp, dz in loader:
            preds.append(model(img.to(device), kp.to(device)).cpu())
            gts.append(dz)
    pred, gt = torch.cat(preds), torch.cat(gts)
    elev = torch.tensor([data.elev[f.stem.rsplit("_", 1)[0]] for f in data.files])
    print(f"検証の奥行き誤差 {(pred - gt).abs().mean().item() * UNIT:.2f} mm")
    print("骨ごとの手前／奥の正解率（30° 以上傾いている骨）: 全体 / カメラがほぼ水平（±20°）")
    for a, b, L in bones:
        g, p = gt[:, b] - gt[:, a], pred[:, b] - pred[:, a]
        big = g.abs() * UNIT > L * 0.5
        low = big & (elev.abs() < 20)
        acc = lambda m: f"{((g[m] > 0) == (p[m] > 0)).float().mean().item() * 100:5.1f}%" if m.sum() else "   —  "
        print(f"  {names[a]:>10s}-{names[b]:<10s} {acc(big)} / {acc(low)}  ({int(big.sum())} / {int(low.sum())} 件)")


def main() -> None:
    p = argparse.ArgumentParser(description="画像からの奥行き推定モデルの学習")
    p.add_argument("--yolo", default=str(REPO / "data" / "synth" / "yolo"))
    p.add_argument("--meta", default=str(REPO / "data" / "synth" / "raw" / "meta"))
    p.add_argument("--size", type=int, default=160, help="切り出し画像の大きさ（px）")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu", help="例: 0（GPU）/ cpu")
    p.add_argument("--no-pretrained", action="store_true", help="ImageNet の事前学習重みを使わない（ダウンロードできないとき）")
    p.add_argument("--out", default=str(REPO / "web" / "models"))
    p.add_argument("--eval-only", metavar="PT", help="学習せず、この重み（depth.pt）を --yolo の val で評価だけする（別生成のテスト用データでの確認用）")
    args = p.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    kcfg = yaml.safe_load((REPO / "configs" / "keypoints.yaml").read_text(encoding="utf-8"))
    names = kcfg["keypoints"]
    idx = {n: i for i, n in enumerate(names)}
    flip = [idx[n.replace("l_", "#").replace("r_", "l_").replace("#", "r_")] for n in names]
    bones = [(idx[a], idx[b], L) for (a, b), L in BONE_MM.items()]

    common = dict(names=names, flip=flip, bones=bones, size=args.size)
    if args.eval_only:
        va = DepthData(Path(args.yolo), Path(args.meta), "val", train=False, **common)
        print(f"評価 {len(va)} 枚（{device}）")
        model = DepthNet(len(names), pretrained=False).to(device)
        model.load_state_dict(torch.load(args.eval_only, map_location=device))
        evaluate(model, DataLoader(va, args.batch, num_workers=args.workers), va, names, bones, device)
        return
    tr = DepthData(Path(args.yolo), Path(args.meta), "train", train=True, **common)
    va = DepthData(Path(args.yolo), Path(args.meta), "val", train=False, **common)
    print(f"学習 {len(tr)} 枚 / 検証 {len(va)} 枚（{device}）")
    tl = DataLoader(tr, args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                    persistent_workers=args.workers > 0)
    vl = DataLoader(va, args.batch, num_workers=args.workers)

    model = DepthNet(len(names), pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=args.epochs * len(tl))
    for ep in range(args.epochs):
        model.train()
        total, n = 0.0, 0
        for i, (img, kp, dz) in enumerate(tl):
            loss = nn.functional.smooth_l1_loss(model(img.to(device), kp.to(device)), dz.to(device), beta=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total, n = total + loss.item() * len(img), n + len(img)
            if i % 100 == 0:
                print(f"\repoch {ep + 1}/{args.epochs}  {i}/{len(tl)}  loss {total / n:.4f}", end="", flush=True)
        print(f"\repoch {ep + 1}/{args.epochs}  loss {total / n:.4f}" + " " * 20)
        torch.save(model.state_dict(), Path(args.out) / "depth.pt")  # 途中経過
    evaluate(model, vl, va, names, bones, device)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.cpu().eval()
    torch.onnx.export(model, (torch.zeros(1, 3, args.size, args.size), torch.zeros(1, 3 * len(names))),
                      str(out / "depth.onnx"), input_names=["image", "kp"], output_names=["dz"], opset_version=17,
                      dynamo=False)
    meta = {"size": args.size, "crop_scale": CROP_SCALE, "unit_mm": UNIT, "keypoints": names, "hips": "hips",
            "bones": [[names[a], names[b], L] for a, b, L in bones]}
    (out / "depth.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"書き出しました: {out / 'depth.onnx'}（{(out / 'depth.onnx').stat().st_size / 1e6:.1f} MB）, depth.json")


if __name__ == "__main__":
    main()
