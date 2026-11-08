# Credits / クレジット

Dummy13 VRM Viewer — Copyright (C) 2026 pt
Licensed under the GNU Affero General Public License v3.0 ([LICENSE](LICENSE)).

This is an unofficial fan tool. It is not made, endorsed or supported by soozafone / Lucky 13 Toys or by VRChat Inc.
非公式のファンツールです。Dummy 13 の作者（soozafone / Lucky 13 Toys）や VRChat の公式ツールではありません。

## Used in this project / このプロジェクトで使っているもの

| What / もの | Copyright / 権利者 | License / ライセンス | Notes / 備考 |
|---|---|---|---|
| [Dummy 13 version 1.0](https://www.printables.com/model/981111-dummy-13-version-10) | soozafone (Lucky 13 Toys) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | STL files used to generate the synthetic training data and the images in `docs/images/`. Not included in this repository. / 合成データと `docs/images/` の画像の元。STL は同梱していない |
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | Ultralytics Inc. | AGPL-3.0 | Training framework and pretrained weights for the pose model. / 姿勢推定モデルの学習と事前学習済み重み |
| [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) | Microsoft Corporation | MIT | Runs the model in the browser (loaded from a CDN). / ブラウザでの推論（CDN から読み込み） |
| [three.js](https://github.com/mrdoob/three.js) | three.js authors | MIT | 3D avatar view (loaded from a CDN). / 3D 表示（CDN から読み込み） |
| [three-vrm](https://github.com/pixiv/three-vrm) | pixiv Inc. | MIT | VRM loading (loaded from a CDN). / VRM の読み込み（CDN から読み込み） |
| [Noto Sans JP](https://fonts.google.com/noto/specimen/Noto+Sans+JP) | The Noto Project Authors | [SIL OFL 1.1](web/fonts/OFL.txt) | Heading font, a Latin-only subset of the Black weight (`web/fonts/`). / 見出しのフォント（Black の英数字だけに絞ったもの） |
| [PyTorch / torchvision](https://github.com/pytorch/vision) | PyTorch contributors | BSD-3-Clause | Depth model (MobileNetV3; ImageNet-pretrained weights). / 奥行きモデル（MobileNetV3、ImageNet の事前学習済み重み） |
| [Blender](https://www.blender.org/) | Blender Foundation | GPL | Tool used to render synthetic data (output images are not covered by the GPL). / 合成データのレンダリングに使う道具 |
| [Poly Haven](https://polyhaven.com/) | Poly Haven contributors | CC0 | Background textures and HDRIs, downloaded by each user (not included). / 背景素材（各自ダウンロード、同梱しない） |

## Trained models / 学習済みモデル

The models in `web/models/` are licensed under the GNU AGPL-3.0, like the code.
`web/models/` の学習済みモデルも、コードと同じく GNU AGPL-3.0 です。

| File / ファイル | What / 中身 | Trained on / 学習データ |
|---|---|---|
| `dummy13-pose.onnx`, `model.json` | Pose model (Ultralytics YOLO pose). / 姿勢推定モデル（Ultralytics YOLO-pose） | Synthetic images rendered from the Dummy 13 STL files (CC BY 4.0) with Poly Haven backgrounds (CC0), plus photos of real figures. / Dummy 13 の STL（CC BY 4.0）から作った合成画像（背景は Poly Haven、CC0）と、実物のフィギュアの写真 |
| `depth.onnx`, `depth.json` | Depth model (MobileNetV3, ImageNet-pretrained). / 奥行きモデル（MobileNetV3、ImageNet の事前学習済み重みから作成） | The same synthetic images. / 同じ合成画像 |

## Avatars / アバター

`web/avatars/pt.vrm` and `web/avatars/panda-san.vrm` are made by pt and distributed under the terms written in the files
([VRM Public License 1.0](https://vrm.dev/licenses/1.0/)): anyone may use, modify and redistribute them, including commercially; no credit is required.
See [web/avatars/README.md](web/avatars/README.md). Use any other VRM avatar according to its own terms of use.

`web/avatars/` の pt・panda-san は作者（pt）の自作で、ファイルに書き込んだ条件（[VRM Public License 1.0](https://vrm.dev/licenses/1.0/)）で配布しています。
商用・改変・再配布を含めて誰でも使えます（クレジット表記は不要）。詳しくは [web/avatars/README.md](web/avatars/README.md)。ほかの VRM は各アバターの利用条件に従って使ってください。
