# GPU のある PC で合成データを量産して学習する

Windows + NVIDIA GPU を想定（Linux でもコマンドはほぼ同じ）。

## 1. 準備（最初の1回だけ）

Blender の部分（リグ作成・合成画像の生成）と、それ以外（背景合成・学習）で Python を分ける。

### Blender：Blender 本体を使う（おすすめ）

https://www.blender.org/download/ から **Blender 4.2 以降**（5.x でも可）をインストールする。
Blender に Python と numpy が入っているので、追加のインストールは不要。
以下、Blender の場所を `BLENDER` とする（Git Bash の例）:

```bash
BLENDER="/c/Program Files/Blender Foundation/Blender 5.0/blender.exe"
"$BLENDER" --version
```

Git Bash を開き直すたびに消えるので、`~/.bashrc` に書いておくと毎回入る（パスは自分の Blender のバージョンに合わせる）:

```bash
echo 'export BLENDER="/c/Program Files/Blender Foundation/Blender 5.0/blender.exe"' >> ~/.bashrc
source ~/.bashrc
```

（`pip install bpy` でも動くが、**Python 3.11 限定**で、numpy も 2 未満に下げる必要がある。）

### それ以外：普通の Python（3.10〜3.13 どれでも可）

Git Bash の場合:

```bash
git clone https://github.com/pt-da-yooon/dummy13-vrm-viewer.git
cd dummy13-vrm-viewer

python -m venv .venv
source .venv/Scripts/activate          # PowerShell なら .venv\Scripts\activate

# PyTorch（GPU 版）を先に入れる。CUDA 12.6 用。ドライバが古ければ cu124 / cu121 に
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

python -c "import torch; print(torch.cuda.is_available())"   # True になれば OK
```

## 2. リグと背景素材を用意

```bash
"$BLENDER" -b -P synth/build_rig.py   # → models/dummy13_rig.blend
python synth/download_backgrounds.py   # → data/backgrounds, data/hdri（約 2.7GB、数分〜十数分）
```

## 3. お試しで 20 枚作って確認

```bash
"$BLENDER" -b models/dummy13_rig.blend -P synth/generate.py -- --count 20 --out data/synth/test
python synth/preview.py data/synth/test/images --out data/preview_test
```

`data/preview_test/` の画像にフィギュアとキーポイントが描かれていれば OK。

- 既定の描画エンジンは EEVEE（速い）。**画像が真っ黒・何も出ない・落ちる**場合は、
  `synth/rig_config.json` の `"render"` を `{"engine": "CYCLES", "resolution": [640, 480], "samples": 16}` にする
  （Cycles は自動で GPU を使う。ログに「Cycles: GPU を使用」と出る）。
- ログの「1 枚あたりの時間」を見て、量産の枚数を決める。

## 4. 量産

```bash
"$BLENDER" -b models/dummy13_rig.blend -P synth/generate.py -- --count 20000
python synth/composite.py --per-image 2      # 背景を 2 通り合成 → 4 万枚
```

時間がかかるので、ウィンドウを複数開いて分割してもよい（同じ `--seed` なら番号ごとに再現性あり）:

```bash
"$BLENDER" -b models/dummy13_rig.blend -P synth/generate.py -- --start 0     --count 10000
"$BLENDER" -b models/dummy13_rig.blend -P synth/generate.py -- --start 10000 --count 10000
```

途中で止めても、`--start` をずらして続きから作れる。

## 5. 学習

```bash
python train/train.py --epochs 100 --batch 32 --device 0
```

- 結果は `runs/pose/dummy13/`。`weights/best.pt` が学習済みの重み。
- GPU のメモリが足りないと言われたら `--batch 16` や `8` に下げる。
- 最後に表示される `pose mAP50-95` が精度の目安（1 に近いほど良い）。

## 困ったとき

- `pip install bpy` が「No matching distribution」→ Python が 3.11 ではない。Blender 本体を使う方が簡単
- `numpy.core.multiarray failed to import`（pip の bpy を使った場合）→ `pip install "numpy<2" "opencv-python<4.12"`

## 6. Webカメラで試す

```bash
python infer/camera_test.py               # 使えるカメラ番号を確認（仮想カメラ VCAM などが 0 番を取ることがある）
python infer/webcam.py --camera 1         # 学習済みの重みで推論。q で終了
python infer/webcam.py --camera 1 --osc   # VRChat に関節角度を送る（試作）
```
