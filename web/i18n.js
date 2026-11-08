// 画面の言語（日本語／English）。ヘッダーの JP / EN で切り替え、選んだものはこのブラウザに覚えておく。
// 最初はブラウザの言語が日本語なら日本語、それ以外は英語。<head> で読み込む。
// HTML は data-i18n="キー"（文字だけ）か data-i18n-html="キー"（リンクなどを含む）で書き換える。
// スクリプトからは window.i18n.t("キー") で今の言語の文を取り、言語が変わると "dummy13-lang" イベントが届く。
(() => {
  const KEY = "dummy13-lang";
  const DICT = {
    ja: {
      sub: "Webカメラに映した Dummy 13 の姿勢を推定します",
      camera: "カメラ", cameraLoading: "（読み込み中）", start: "カメラ開始", tryFile: "画像・動画で試す", pickModel: "モデル（.onnx）を選ぶ",
      avatar: "アバター", angles: "関節の角度",
      pointsLegend: '点：<b class="c-l">左</b> / <b class="c-r">右</b> / <b class="c-c">胴</b>（Dummy13 から見た左右）',
      threshold: "しきい値", smooth: "平滑化", mirror: "左右反転", bones: "骨格を表示",
      mannequin: "マネキン", loadVrm: "VRM を読み込む", resetView: "視点を戻す", useDepth: "奥行きを推定",
      overlayMode: "カメラ画像に重ねる", lightBg: "背景を明るく",
      confHigh: "信頼度が高い", confLow: "低い",
      privacy: "処理はお使いの PC・スマホなどのブラウザ内で行います。カメラの映像や読み込んだ VRM は外部に送信せず、学習もしません。オフラインでも動きます。",
      credits: 'Dummy 13 の非公式のファンツールです · 説明：<a href="https://protopedia.net/prototype/10032" target="_blank" rel="noopener">ProtoPedia</a> / <a href="https://www.youtube.com/watch?v=VyDbEDj0Njs" target="_blank" rel="noopener">YouTube</a> · <a href="https://github.com/pt-da-yooon/dummy13-vrm-viewer" target="_blank" rel="noopener">ソースコード</a><br>クレジット：Dummy 13 by <a href="https://www.printables.com/model/981111-dummy-13-version-10" target="_blank" rel="noopener">soozafone</a>（CC BY 4.0） · 姿勢推定：Ultralytics YOLO（AGPL-3.0）',
      speed: "速度", detect: "検出", loadingShort: "読込中", none: "なし",
      cameraN: "カメラ {n}", noCamera: "カメラが見つかりません", cameraUnavailable: "カメラを使えません",
      loadingModel: "モデルを読み込んでいます…", ready: "「カメラ開始」を押すか、画像・動画を選んでください",
      modelFailed: "モデルを読み込めませんでした", cameraError: "カメラを開けません: {msg}",
      modelMissing: "モデル（dummy13-pose.onnx）が見つかりません。上の「モデルを選ぶ」から .onnx を指定してください。",
      vrmFailed: "読み込めませんでした: {msg}", notVrm: "VRM ではありません",
      l_elbow: "左ひじ", r_elbow: "右ひじ", l_knee: "左ひざ", r_knee: "右ひざ",
      l_shoulder: "左肩", r_shoulder: "右肩", l_hip: "左股", r_hip: "右股",
      hideUi: "ボタンを隠す（H）",
      captureHelp: "ドラッグで回す ・ ホイールで寄る／離れる ・ 右ドラッグで平行移動 ・ H でボタンの表示／非表示 ・ F11 で全画面",
    },
    en: {
      sub: "Estimates the pose of a Dummy 13 figure from your webcam",
      camera: "Camera", cameraLoading: "(loading…)", start: "Start camera", tryFile: "Try an image or video", pickModel: "Choose a model (.onnx)",
      avatar: "Avatar", angles: "Joint angles",
      pointsLegend: 'Points: <b class="c-l">left</b> / <b class="c-r">right</b> / <b class="c-c">torso</b> (as seen from Dummy13)',
      threshold: "Threshold", smooth: "Smoothing", mirror: "Mirror", bones: "Show skeleton",
      mannequin: "Mannequin", loadVrm: "Load VRM", resetView: "Reset view", useDepth: "Estimate depth",
      overlayMode: "Overlay on camera", lightBg: "Light background",
      confHigh: "High confidence", confLow: "Low",
      privacy: "Everything runs in the browser on your own PC or phone. Your camera video and the VRM files you load are never sent anywhere and are never used for training. It also works offline.",
      credits: 'An unofficial fan tool for Dummy 13 · About (Japanese): <a href="https://protopedia.net/prototype/10032" target="_blank" rel="noopener">ProtoPedia</a> / <a href="https://www.youtube.com/watch?v=VyDbEDj0Njs" target="_blank" rel="noopener">YouTube</a> · <a href="https://github.com/pt-da-yooon/dummy13-vrm-viewer" target="_blank" rel="noopener">Source code</a><br>Credits: Dummy 13 by <a href="https://www.printables.com/model/981111-dummy-13-version-10" target="_blank" rel="noopener">soozafone</a> (CC BY 4.0) · Pose model: Ultralytics YOLO (AGPL-3.0)',
      speed: "Speed", detect: "Detection", loadingShort: "Loading", none: "None",
      cameraN: "Camera {n}", noCamera: "No camera found", cameraUnavailable: "Camera unavailable",
      loadingModel: "Loading the model…", ready: "Press “Start camera” or choose an image or video",
      modelFailed: "Couldn't load the model", cameraError: "Can't open the camera: {msg}",
      modelMissing: "The model (dummy13-pose.onnx) was not found. Choose a .onnx file with “Choose a model” above.",
      vrmFailed: "Couldn't load: {msg}", notVrm: "Not a VRM file",
      l_elbow: "Left elbow", r_elbow: "Right elbow", l_knee: "Left knee", r_knee: "Right knee",
      l_shoulder: "Left shoulder", r_shoulder: "Right shoulder", l_hip: "Left hip", r_hip: "Right hip",
      hideUi: "Hide buttons (H)",
      captureHelp: "Drag to rotate · Scroll to zoom · Right-drag to pan · H to show/hide the buttons · F11 for full screen",
    },
  };
  const read = () => { try { return localStorage.getItem(KEY); } catch { return null; } };
  const save = (l) => { try { localStorage.setItem(KEY, l); } catch { /* 保存できなくても切り替えはできる */ } };
  let lang = read() || ((navigator.language || "").toLowerCase().startsWith("ja") ? "ja" : "en");
  if (!DICT[lang]) lang = "ja";

  const t = (key, vars = {}) => (DICT[lang][key] ?? DICT.ja[key] ?? key).replace(/\{(\w+)\}/g, (_, k) => vars[k] ?? "");
  function apply() {
    document.documentElement.lang = lang;
    for (const el of document.querySelectorAll("[data-i18n]")) el.textContent = t(el.dataset.i18n);
    for (const el of document.querySelectorAll("[data-i18n-html]")) el.innerHTML = t(el.dataset.i18nHtml);
    for (const b of document.querySelectorAll("[data-lang]")) b.setAttribute("aria-pressed", String(b.dataset.lang === lang));
  }
  function set(l) {
    if (!DICT[l] || l === lang) return;
    lang = l; save(l); apply();
    window.dispatchEvent(new CustomEvent("dummy13-lang", { detail: lang }));
  }
  window.i18n = { t, set, get lang() { return lang; } };
  document.documentElement.lang = lang;
  document.addEventListener("DOMContentLoaded", () => {
    apply();
    for (const b of document.querySelectorAll("[data-lang]")) b.addEventListener("click", () => set(b.dataset.lang));
  });
})();
