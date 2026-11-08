"""2D キーポイント → 奥行き（各関節の手前／奥）を推定する小さなモデルを学習する。

骨の長さだけでは「手前に傾いているのか奥に傾いているのか」が分からない
（例：後ろに反らせても前に倒したように見える）。合成データの meta に入っている
3D の正解（keypoints_cam）を使い、全身の 2D の形から各関節の奥行きを当てる MLP を学習する。
レンダリング画像は使わないので、generate.py の --dry-run で作った meta だけでも学習できる。

使い方:
    python train/train_lifter.py                          # data/synth/raw/meta を使う
    python train/train_lifter.py --meta path/to/meta --epochs 40
出力: web/models/lifter.json（ブラウザ版の avatar.js が読み込む）
"""

import argparse
import base64
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

REPO = Path(__file__).resolve().parent.parent

# web/avatar.js の BONE_MM と同じ（Dummy13 の骨の長さ、mm）。入力の尺度合わせに使う。
BONE_MM = {
    ("hips", "chest"): 34.0, ("chest", "neck"): 10.0, ("neck", "head"): 14.5,
    ("neck", "l_shoulder"): 14.32, ("l_shoulder", "l_elbow"): 22.2, ("l_elbow", "l_wrist"): 18.0,
    ("neck", "r_shoulder"): 14.32, ("r_shoulder", "r_elbow"): 22.2, ("r_elbow", "r_wrist"): 18.0,
    ("hips", "l_hip"): 8.0, ("l_hip", "l_knee"): 30.2, ("l_knee", "l_ankle"): 31.0,
    ("hips", "r_hip"): 8.0, ("r_hip", "r_knee"): 30.2, ("r_knee", "r_ankle"): 31.0,
}
UNIT = 100.0  # 入出力は mm / 100


def load(meta_dir: Path, names: list[str]) -> np.ndarray:
    """meta から (N, K, 3) のカメラ座標（x 右, y 下, z 奥; m）を読む。"""
    out = []
    for f in sorted(meta_dir.glob("*.json")):
        kc = json.loads(f.read_text(encoding="utf-8"))["keypoints_cam"]
        out.append([kc[n] for n in names])
    return np.asarray(out, np.float32)


def make_inputs(uv: torch.Tensor, mask: torch.Tensor, bones, hips: int) -> torch.Tensor:
    """2D（y 上向き）と使える点の印 → モデルの入力。avatar.js の lifterInput と同じ計算。

    腰を原点にし、「画面上の骨の長さ ÷ 本来の長さ」の最大値で割って mm 相当にそろえる。
    """
    rel = uv - uv[:, hips:hips + 1]
    s = torch.zeros(len(uv))
    for a, b, L in bones:
        ok = mask[:, a] * mask[:, b]
        s = torch.maximum(s, ok * (uv[:, b] - uv[:, a]).norm(dim=1) / L)
    s = s.clamp_min(1e-9)
    x = rel / s[:, None, None] / UNIT * mask[..., None]
    return torch.cat([x.flatten(1), mask], dim=1)


class Lifter(nn.Module):
    def __init__(self, k: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * k, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, k),
        )

    def forward(self, x):
        return self.net(x)


def batch(P: torch.Tensor, flip: list[int], hips: int, bones, rng: torch.Generator, train: bool):
    """3D の正解から（入力, 奥行きの正解, 使える点の印）を作る。学習時は左右反転・欠け・ノイズを足す。"""
    P = P.clone()
    if train:
        m = torch.rand(len(P), generator=rng) < 0.5  # 左右反転（x を反転して左右の名前を入れ替える）
        Pf = P[:, flip].clone()
        Pf[..., 0] *= -1
        P[m] = Pf[m]
    uv = P[..., :2] / P[..., 2:3]
    uv[..., 1] *= -1  # y 上向き
    dz = -(P[..., 2] - P[:, hips:hips + 1, 2]) * 1000 / UNIT  # 手前が +（mm/100）
    mask = torch.ones(P.shape[:2])
    if train:
        mask = (torch.rand(P.shape[:2], generator=rng) > 0.08).float()  # 信頼度が低くて使えない点
        mask[:, hips] = 1
        scale = (uv - uv[:, hips:hips + 1]).norm(dim=2).amax(dim=1)
        uv = uv + torch.randn(uv.shape, generator=rng) * scale[:, None, None] * 0.015
    return make_inputs(uv, mask, bones, hips), dz, mask


def main() -> None:
    p = argparse.ArgumentParser(description="2D→奥行き推定モデル（lifter）の学習")
    p.add_argument("--meta", default=str(REPO / "data" / "synth" / "raw" / "meta"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--out", default=str(REPO / "web" / "models" / "lifter.json"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    kcfg = yaml.safe_load((REPO / "configs" / "keypoints.yaml").read_text(encoding="utf-8"))
    names = kcfg["keypoints"]
    idx = {n: i for i, n in enumerate(names)}
    flip = [idx[n.replace("l_", "#").replace("r_", "l_").replace("#", "r_")] for n in names]
    bones = [(idx[a], idx[b], L) for (a, b), L in BONE_MM.items()]
    hips = idx["hips"]

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    data = torch.from_numpy(load(Path(args.meta), names))
    if len(data) < 100:
        raise SystemExit(f"{args.meta} の meta が少なすぎます（{len(data)} 件）")
    perm = torch.randperm(len(data))
    n_val = max(len(data) // 10, 1)
    val, tr = data[perm[:n_val]], data[perm[n_val:]]
    print(f"学習 {len(tr)} 件 / 検証 {len(val)} 件")

    model = Lifter(len(names))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    rng = torch.Generator().manual_seed(args.seed)
    xv, dzv, _ = batch(val, flip, hips, bones, rng, train=False)
    for ep in range(args.epochs):
        model.train()
        order = torch.randperm(len(tr))
        total = 0.0
        for i in range(0, len(tr), 256):
            x, dz, _ = batch(tr[order[i:i + 256]], flip, hips, bones, rng, train=True)
            loss = nn.functional.smooth_l1_loss(model(x), dz, beta=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        sched.step()
        model.eval()
        with torch.no_grad():
            err = (model(xv) - dzv).abs().mean().item() * UNIT
        print(f"epoch {ep + 1:3d}/{args.epochs}  loss {total / len(tr):.4f}  検証の奥行き誤差 {err:.2f} mm")

    # 骨ごとの「手前／奥」の正解率（骨が十分に傾いているものだけ）
    with torch.no_grad():
        pred = model(xv)
    # 比べる相手：これまでの決め打ち（avatar.js の TREE。腕・太もも・胴は体の前、すねは後ろ）
    uv = val[..., 0] / val[..., 2]
    lr = (uv[:, idx["r_shoulder"]] - uv[:, idx["l_shoulder"]]) + (uv[:, idx["r_hip"]] - uv[:, idx["l_hip"]])
    facing = torch.where(lr >= 0, 1.0, -1.0)  # x 右向きの画像で、体の左が右にある＝こちらを向いている
    rule = {"chest": 1, "neck": 1, "head": 1, "l_elbow": 1, "l_wrist": 1, "r_elbow": 1, "r_wrist": 1,
            "l_knee": 1, "r_knee": 1, "l_ankle": -1, "r_ankle": -1}
    print("骨ごとの手前／奥の正解率（30° 以上傾いている骨）:  モデル / これまでの決め打ち")
    accs = []
    for a, b, L in bones:
        gt, pr = dzv[:, b] - dzv[:, a], pred[:, b] - pred[:, a]
        big = gt.abs() * UNIT > L * 0.5
        if big.sum() == 0:
            continue
        acc = ((gt[big] > 0) == (pr[big] > 0)).float().mean().item()
        accs.append(acc)
        r = rule.get(names[b])
        old = f"{((gt[big] > 0) == (facing[big] * r > 0)).float().mean().item() * 100:5.1f}%" if r else "  （0 扱い）"
        print(f"  {names[a]:>10s}-{names[b]:<10s} {acc * 100:5.1f}% / {old}  ({int(big.sum())} 件)")
    print(f"  平均 {np.mean(accs) * 100:.1f}%")

    layers = [m for m in model.net if isinstance(m, nn.Linear)]
    enc = lambda t: base64.b64encode(t.detach().numpy().astype("<f4").tobytes()).decode()
    out = {
        "keypoints": names, "unit_mm": UNIT, "hips": "hips",
        "bones": [[names[a], names[b], L] for a, b, L in bones],
        "layers": [{"in": l.in_features, "out": l.out_features, "w": enc(l.weight), "b": enc(l.bias)} for l in layers],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"書き出しました: {args.out}（{Path(args.out).stat().st_size / 1e6:.1f} MB）")


if __name__ == "__main__":
    main()
