# web/ — ブラウザ版 Dummy13 VRM Viewer

学習済みモデル（ONNX）を ONNX Runtime Web でブラウザの中だけで動かすデモ。
映像はどこにも送らない。WebGPU が使えるブラウザ（Chrome / Edge など）では GPU、それ以外は CPU（WebAssembly）で動く。

## 手元で動かす

```bash
python train/export_web.py                 # runs/pose/dummy13/weights/best.pt → web/models/
cd web
python -m http.server 8000                 # http://localhost:8000 を開く
```

- `index.html` をダブルクリックで開く（file://）と動かない。必ず上のように簡易サーバー経由で開く。
- `web/models/dummy13-pose.onnx` が無いときは、画面の「モデルを選ぶ」から .onnx を指定できる。
- カメラは「カメラ開始」、画像・動画ファイルでも試せる。
- 右側に 3D アバターを表示し、推定したポーズで動かす（`avatar.js`、three.js + three-vrm）。
  「背景を明るく」で 3D 表示の背景だけを明るい灰色にできる（選んだものはブラウザに覚えておき、撮影用ページも同じ設定で開く）。
  既定は簡単なマネキン。「VRM を読み込む」で自分のアバター（VRM 0.x / 1.0）に差し替えられる。
  「奥行きを推定」をオンにすると、骨の長さと画像上の長さの差から手前・奥への曲がりも付ける（前後の向きは外れることがある）。
  「カメラ画像に重ねる」をオンにすると、カメラ画像の上の Dummy13 と同じ位置・大きさにアバターを重ねて描く
  （正射影。大きさは胴の長さ＝首〜腰で合わせる）。
- 奥行き（手前／奥）は、次の順で使えるものを使う。
  1. `models/depth.onnx` + `depth.json`（`train/train_depth.py`）：切り出した画像と 2D キーポイントから推定。後ろに反らす／前に倒すの区別が付く
  2. `models/lifter.json`（`train/train_lifter.py`）：2D の形だけから推定（軽い。画像の手がかりは使わない）
  3. どちらも無ければ骨の長さと決め打ちの向き（胴・腕・太ももは前、すねは後ろ）
- 読み込めるアバターは **VRM のみ**（FBX には対応しない。ボーンの対応付けが無く、Unity のシェーダーも失われるため）。
  VRChat 用アバターは、BOOTH などで配られている **VRM 版**を使うか、Unity の UniVRM で VRM に書き出す
  （変換が利用規約で認められているか確認すること）。

## 撮影用ページ（capture.html）

`http://localhost:8000/capture.html` を開くと、アバターの 3D 表示だけが画面いっぱいに出る（動画の撮影用）。
「VRM を読み込む」で VRM を選び、ドラッグで回す・ホイールで寄る／離れる・右ドラッグで平行移動。
H キーでボタンと説明を隠せる。F11 で全画面にして、OBS などで画面を録画する。

## 点付けツール（label.html）

実物の写真に関節の点を付けて、追加学習用のラベルを作る（Chrome / Edge）。`http://localhost:8000/label.html` で開く。

1. `data/real/images/` に写真を入れる（または画面の「カメラで撮る」で撮影すると、そこに保存される）
2. 「フォルダを開く」で `data/real` を選ぶ（中に `images/` と `labels/` ができる）
3. 今のモデルが下書きした点をドラッグで直し、Enter で保存して次へ（ラベルは `labels/<写真の名前>.txt`、YOLO-pose 形式）
4. 学習に混ぜる：`python train/train.py --model runs/pose/dummy13c/weights/best.pt --name dummy13r --epochs 10 --device 0 --batch 8 --real data/real`
   （実物は 10 回繰り返して水増し。評価は実物の写真の 2 割で行うので、実物での精度が分かる。`--synth-fraction 0.3` で合成を減らすと速い）

点の状態：見えている（2）／隠れている（1、位置は推測して置く）／写っていない（0）。合成データと同じ決め方。

## 公開するとき

`web/` をそのまま GitHub Pages などの静的サイトに置けばよい（サーバー側の処理は無い）。
このリポジトリでは `.github/workflows/pages.yml` が、main に push するたびに `web/` を GitHub Pages に公開する（Settings → Pages の Source を「GitHub Actions」にしておく）。
学習済みモデル（`models/`、姿勢推定 約 12MB・奥行き 約 14MB）はリポジトリに含めているので、そのまま公開できる。自分で学習したモデルに差し替えるときは `train/export_web.py` で書き出す。

## 構成

- `index.html` / `style.css` / `app.js` — 画面と推論（ビルド不要）
- `sw.js` — オフライン用の Service Worker。一度開いたときにページ・モデル・CDN のライブラリを保存し、通信できないときはそれを使う（https か localhost のときだけ動く）
- `ogp.png` — SNS にリンクを貼ったときのカードの画像（1200×630。index.html の OGP・X のカードで使う。URL は公開先に合わせて書いてある）
- `i18n.js` — 画面の言語（日本語／English）。ヘッダーの JP / EN で切り替え、選んだものはブラウザに覚えておく（最初はブラウザの言語に合わせる）。撮影用ページも同じ設定に従う
- `avatar.js` — 3D アバターの表示とポーズの反映（ボーンごとに、休止姿勢での向き → 推定した向き への回転をかける）
- `models/model.json` — キーポイント名・骨格・入力サイズ（`train/export_web.py` が書き出す）
- 前処理はレターボックス（灰色 114 で余白を埋めて正方形に）、出力 `[1, 4+1+16×3, N]` から最も信頼度の高い1体を使う。
