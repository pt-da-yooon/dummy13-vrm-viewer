// Dummy13 Pose — ブラウザ内で YOLO-pose（ONNX）を動かして Dummy 13 の関節を推定する。
// 推論は ONNX Runtime Web（WebGPU → WebAssembly の順に試す）。映像は外部に送らない。
"use strict";

const ORT_VERSION = "1.30.0";
const MODEL_URL = "models/dummy13-pose.onnx";
const META_URL = "models/model.json";

// model.json が無いときの既定値（configs/keypoints.yaml と同じ順番）
const DEFAULT_META = {
  imgsz: 640,
  keypoints: ["head", "neck", "chest", "hips", "l_shoulder", "l_elbow", "l_wrist", "r_shoulder", "r_elbow",
              "r_wrist", "l_hip", "l_knee", "l_ankle", "r_hip", "r_knee", "r_ankle"],
  skeleton: [["head", "neck"], ["neck", "chest"], ["chest", "hips"], ["neck", "l_shoulder"],
             ["l_shoulder", "l_elbow"], ["l_elbow", "l_wrist"], ["neck", "r_shoulder"], ["r_shoulder", "r_elbow"],
             ["r_elbow", "r_wrist"], ["hips", "l_hip"], ["l_hip", "l_knee"], ["l_knee", "l_ankle"],
             ["hips", "r_hip"], ["r_hip", "r_knee"], ["r_knee", "r_ankle"]],
};

// 2D の関節角度: 名前 → [端点 a, 関節, 端点 b]
// 表示名は i18n.js（キーは関節の名前。l_ は左、r_ は右）
const ANGLES = {
  l_elbow: ["l_shoulder", "l_elbow", "l_wrist"],
  r_elbow: ["r_shoulder", "r_elbow", "r_wrist"],
  l_knee: ["l_hip", "l_knee", "l_ankle"],
  r_knee: ["r_hip", "r_knee", "r_ankle"],
  l_shoulder: ["neck", "l_shoulder", "l_elbow"],
  r_shoulder: ["neck", "r_shoulder", "r_elbow"],
  l_hip: ["hips", "l_hip", "l_knee"],
  r_hip: ["hips", "r_hip", "r_knee"],
};
const t = (key, vars) => window.i18n.t(key, vars);
// 画面中央のお知らせ。言語を切り替えたら書き直せるように、キーを覚えておく
let overlayMsg = null;
function setOverlay(key, vars) {
  overlayMsg = key ? [key, vars] : null;
  overlay.textContent = key ? t(key, vars) : "";
}
const KPT_CONF = 0.5;  // これ未満のキーポイントは「低い」扱い

const $ = (id) => document.getElementById(id);
const canvas = $("canvas"), ctx = canvas.getContext("2d");
const video = $("video");
const overlay = $("overlay");

let session = null, backend = "", meta = DEFAULT_META, idx = {};
let source = null;       // 推論する要素（video か image）
let running = false, busy = false;
let lastResult = null;   // テスト・デバッグ用に window からも見えるようにする
let fps = 0, lastT = performance.now();

// ---------------------------------------------------------------- One Euro フィルタ（ジッター除去）

class OneEuro {
  constructor(minCutoff = 1.0, beta = 0.05, dCutoff = 1.0) {
    Object.assign(this, { minCutoff, beta, dCutoff, x: null, dx: null, t: 0 });
  }
  static alpha(cutoff, dt) { const r = 2 * Math.PI * cutoff * dt; return r / (r + 1); }
  filter(x, t) {
    if (this.x === null) { this.x = Float32Array.from(x); this.dx = new Float32Array(x.length); this.t = t; return this.x; }
    const dt = Math.max(t - this.t, 1e-3);
    const ad = OneEuro.alpha(this.dCutoff, dt);
    for (let i = 0; i < x.length; i++) {
      const dx = (x[i] - this.x[i]) / dt;
      this.dx[i] += ad * (dx - this.dx[i]);
      const a = OneEuro.alpha(this.minCutoff + this.beta * Math.abs(this.dx[i]), dt);
      this.x[i] += a * (x[i] - this.x[i]);
    }
    this.t = t;
    return this.x;
  }
  reset() { this.x = null; }
}
const smoother = new OneEuro();

// ---------------------------------------------------------------- モデルの読み込み

/** モデルのファイル（.onnx）を取ってきてバイト列にする。
 *  - 1 回だけ取る（WebGPU で失敗して WebAssembly でやり直すときも取り直さない）
 *  - 前に保存したものがあれば、サーバーには「変わったか」だけを聞き、変わっていなければ保存したものを使う（毎回 10MB 以上を取り直さない）
 *  - 通信できなければ、保存したものを使う（オフライン）
 *  保存先は sw.js と同じ置き場。sw.js がまだ働いていない最初の訪問でもここで保存するので、二重に取らずに済む */
const MODEL_CACHE = "dummy13-viewer-v1";
async function fetchModel(url) {
  const cache = window.caches ? await caches.open(MODEL_CACHE).catch(() => null) : null;
  const saved = cache ? await cache.match(url) : null;
  const headers = {};
  if (saved) {
    const etag = saved.headers.get("ETag"), modified = saved.headers.get("Last-Modified");
    if (etag) headers["If-None-Match"] = etag; else if (modified) headers["If-Modified-Since"] = modified;
  }
  let res;
  try {
    res = await fetch(url, { cache: "no-store", headers });
  } catch (e) {
    if (saved) return new Uint8Array(await saved.arrayBuffer());
    throw e;
  }
  if (res.status === 304 && saved) return new Uint8Array(await saved.arrayBuffer());
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  if (cache) cache.put(url, res.clone()).catch(() => {});
  return new Uint8Array(await res.arrayBuffer());
}

async function loadModel(bufferOrUrl) {
  overlay.hidden = false;
  setOverlay("loadingModel");
  if (typeof bufferOrUrl === "string") bufferOrUrl = await fetchModel(bufferOrUrl);
  ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  const tries = navigator.gpu ? ["webgpu", "wasm"] : ["wasm"];
  for (const ep of tries) {
    try {
      session = await ort.InferenceSession.create(bufferOrUrl, { executionProviders: [ep], graphOptimizationLevel: "all" });
      backend = ep === "webgpu" ? "WebGPU（GPU）" : "WebAssembly（CPU）";
      break;
    } catch (e) {
      console.warn(`${ep} で読み込めませんでした`, e);
    }
  }
  if (!session) throw new Error(t("modelFailed"));
  setOverlay("ready");
  setStatus();
  updateAngles(null);  // 推定する前から関節の一覧を「—」で出しておく
}

// 画像から奥行きを当てるモデル（train/train_depth.py が書き出す。無ければ使わない）
let depth = null;  // { session, meta }
async function loadDepth() {
  try {
    const r = await fetch("models/depth.json");
    if (!r.ok) return;
    const dmeta = await r.json();
    const bytes = await fetchModel("models/depth.onnx");
    const tries = navigator.gpu ? ["webgpu", "wasm"] : ["wasm"];
    for (const ep of tries) {
      try {
        depth = { session: await ort.InferenceSession.create(bytes, { executionProviders: [ep] }), meta: dmeta };
        console.log(`奥行き推定モデルを読み込みました（${ep}）`);
        return;
      } catch (e) { console.warn(`奥行きモデル: ${ep} で読み込めませんでした`, e); }
    }
  } catch (_) { /* 無ければ使わない */ }
}

const dcrop = document.createElement("canvas");
const dcropCtx = dcrop.getContext("2d", { willReadFrequently: true });

/** 検出枠の中心から正方形に切り出す（train_depth.py の crop_square と同じ。はみ出しは灰色 114） */
/** 枠の中心から一辺 side の正方形を切り出して S×S の canvas に描く（はみ出しは灰色 114）。戻り値は切り出しの左上 */
function drawSquareCrop(c, cctx, el, w, h, box, side, S) {
  const sx = (box[0] + box[2]) / 2 - side / 2, sy = (box[1] + box[3]) / 2 - side / 2;
  c.width = S; c.height = S;
  cctx.fillStyle = "rgb(114,114,114)";
  cctx.fillRect(0, 0, S, S);
  // 画像の中に入っている部分だけを、切り出し先の対応する位置に描く
  const ix0 = Math.max(sx, 0), iy0 = Math.max(sy, 0), ix1 = Math.min(sx + side, w), iy1 = Math.min(sy + side, h);
  if (ix1 > ix0 && iy1 > iy0) {
    const k = S / side;
    cctx.drawImage(el, ix0, iy0, ix1 - ix0, iy1 - iy0, (ix0 - sx) * k, (iy0 - sy) * k, (ix1 - ix0) * k, (iy1 - iy0) * k);
  }
  return [sx, sy];
}

function cropForDepth(el, w, h, box) {
  const S = depth.meta.size;
  const side = Math.max(box[2] - box[0], box[3] - box[1]) * depth.meta.crop_scale;
  drawSquareCrop(dcrop, dcropCtx, el, w, h, box, side, S);
  const px = dcropCtx.getImageData(0, 0, S, S).data;
  const out = new Float32Array(3 * S * S);
  for (let i = 0, p = 0; i < S * S; i++, p += 4) {
    out[i] = px[p] / 255; out[i + S * S] = px[p + 1] / 255; out[i + 2 * S * S] = px[p + 2] / 255;
  }
  return new ort.Tensor("float32", out, [1, 3, S, S]);
}

/** キーポイント側の入力（train_lifter.py の make_inputs と同じ。y は上向き） */
function depthKpInput(xy, kc) {
  const names = depth.meta.keypoints, K = names.length, U = depth.meta.unit_mm;
  const id = (n) => idx[n];
  const ok = (n) => kc[id(n)] >= KPT_CONF;
  const p = (n) => [xy[2 * id(n)], -xy[2 * id(n) + 1]];
  let s = 0;
  for (const [a, b, L] of depth.meta.bones) {
    if (!ok(a) || !ok(b)) continue;
    const [ax, ay] = p(a), [bx, by] = p(b);
    s = Math.max(s, Math.hypot(bx - ax, by - ay) / L);
  }
  const x = new Float32Array(3 * K);
  const [hx, hy] = p(depth.meta.hips);
  names.forEach((n, i) => {
    if (!ok(n) || s <= 0) return;
    const [px, py] = p(n);
    x[2 * i] = (px - hx) / s / U; x[2 * i + 1] = (py - hy) / s / U; x[2 * K + i] = 1;
  });
  return new ort.Tensor("float32", x, [1, 3 * K]);
}

/** 各関節の奥行き（腰から手前が +、mm）。名前 → 値 */
async function inferDepth(el, w, h, res) {
  const out = await depth.session.run({ image: cropForDepth(el, w, h, res.box), kp: depthKpInput(res.xy, res.kc) });
  const d = out.dz.data, U = depth.meta.unit_mm;
  return Object.fromEntries(depth.meta.keypoints.map((n, i) => [n, d[i] * U]));
}

async function loadMeta() {
  try {
    const r = await fetch(META_URL);
    if (r.ok) meta = { ...DEFAULT_META, ...(await r.json()) };
  } catch (_) { /* 既定値を使う */ }
  idx = Object.fromEntries(meta.keypoints.map((n, i) => [n, i]));
}

// ---------------------------------------------------------------- 前処理・後処理

const prep = document.createElement("canvas");
const prepCtx = prep.getContext("2d", { willReadFrequently: true });

/** レターボックス（縦横比を保って縮小し、余白を灰色 114 で埋める）して NCHW の Float32 にする */
function preprocess(el, w, h) {
  const S = meta.imgsz;
  prep.width = S; prep.height = S;
  const scale = Math.min(S / w, S / h);
  const nw = Math.round(w * scale), nh = Math.round(h * scale);
  const padX = Math.floor((S - nw) / 2), padY = Math.floor((S - nh) / 2);
  prepCtx.fillStyle = "rgb(114,114,114)";
  prepCtx.fillRect(0, 0, S, S);
  prepCtx.drawImage(el, 0, 0, w, h, padX, padY, nw, nh);
  const px = prepCtx.getImageData(0, 0, S, S).data;
  const out = new Float32Array(3 * S * S);
  for (let i = 0, p = 0; i < S * S; i++, p += 4) {
    out[i] = px[p] / 255;
    out[i + S * S] = px[p + 1] / 255;
    out[i + 2 * S * S] = px[p + 2] / 255;
  }
  return { tensor: new ort.Tensor("float32", out, [1, 3, S, S]), scale, padX, padY };
}

/** 出力 [1, 4+1+K*3, N] から一番自信のある1体を取り出す（Dummy13 は1体だけ映す想定） */
function postprocess(output, { scale, padX, padY }, confThr) {
  const [, C, N] = output.dims;
  const d = output.data;
  let best = -1, bestConf = confThr;
  for (let i = 0; i < N; i++) {
    const c = d[4 * N + i];
    if (c > bestConf) { bestConf = c; best = i; }
  }
  if (best < 0) return null;
  const K = (C - 5) / 3;
  const unmap = (v, pad) => (v - pad) / scale;
  const cx = d[best], cy = d[N + best], bw = d[2 * N + best], bh = d[3 * N + best];
  const box = [unmap(cx - bw / 2, padX), unmap(cy - bh / 2, padY), unmap(cx + bw / 2, padX), unmap(cy + bh / 2, padY)];
  const xy = new Float32Array(K * 2), kc = new Float32Array(K);
  for (let k = 0; k < K; k++) {
    xy[2 * k] = unmap(d[(5 + 3 * k) * N + best], padX);
    xy[2 * k + 1] = unmap(d[(6 + 3 * k) * N + best], padY);
    kc[k] = d[(7 + 3 * k) * N + best];
  }
  return { conf: bestConf, box, xy, kc };
}

function angle(xy, a, b, c) {
  const v1x = xy[2 * a] - xy[2 * b], v1y = xy[2 * a + 1] - xy[2 * b + 1];
  const v2x = xy[2 * c] - xy[2 * b], v2y = xy[2 * c + 1] - xy[2 * b + 1];
  const n = Math.hypot(v1x, v1y) * Math.hypot(v2x, v2y);
  if (n < 1e-6) return NaN;
  return Math.acos(Math.max(-1, Math.min(1, (v1x * v2x + v1y * v2y) / n))) * 180 / Math.PI;
}

// ---------------------------------------------------------------- 描画

// 骨の色：Dummy13 から見た左＝水色、右＝ピンク、胴＝白（style.css の --left / --right / --center と同じ）。
// フィギュアの色（オレンジ・緑など）と重ならない色にして、さらに黒い縁取りを付ける
const SIDE_COLOR = { l: "#35c8ff", r: "#ff4fd8", c: "#f4f6fa" };
const sideOf = (name) => (name.startsWith("l_") ? "l" : name.startsWith("r_") ? "r" : "c");

function draw(el, w, h, res) {
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h;
    canvas.style.setProperty("--ar", String(w / h));  // 表示の大きさは style.css で、この縦横比のまま枠に収める
  }
  const mirror = $("mirror").checked;
  ctx.save();
  if (mirror) { ctx.translate(w, 0); ctx.scale(-1, 1); }
  ctx.drawImage(el, 0, 0, w, h);
  if (res && $("bones").checked) {  // 「骨格を表示」がオフなら、カメラ画像だけを描く（推定とアバターは動いたまま）
    // 線の太さは画面に表示される大きさで決める（縦長の動画やスマホでは画像が縮んで表示されるので、画素数だけで決めると細くなる）
    const scale = w / (canvas.clientWidth || w);
    const lw = Math.max(2, w / 260, 2.6 * scale);
    const P = (k) => [res.xy[2 * k], res.xy[2 * k + 1]];
    // 枠：四隅だけのカギ括弧
    const [x0, y0, x1, y1] = res.box, c = Math.min(x1 - x0, y1 - y0) * 0.12;
    ctx.strokeStyle = "rgba(180,214,74,0.9)";
    ctx.lineWidth = Math.max(1.5, lw * 0.6);
    ctx.beginPath();
    for (const [x, y, dx, dy] of [[x0, y0, 1, 1], [x1, y0, -1, 1], [x0, y1, 1, -1], [x1, y1, -1, -1]]) {
      ctx.moveTo(x + dx * c, y); ctx.lineTo(x, y); ctx.lineTo(x, y + dy * c);
    }
    ctx.stroke();
    // 骨：黒い縁取り → 光って見える太い半透明の線 → 細い線 の順に重ねる
    ctx.lineCap = "round";
    for (const pass of [0, 1, 2]) {
      for (const [a, b] of meta.skeleton) {
        const i = idx[a], j = idx[b];
        if (res.kc[i] < KPT_CONF || res.kc[j] < KPT_CONF) continue;
        const col = SIDE_COLOR[sideOf(sideOf(b) !== "c" ? b : a)];
        ctx.strokeStyle = pass === 0 ? "#000" : col;
        ctx.globalAlpha = [0.55, 0.3, 1][pass];
        ctx.lineWidth = [lw * 2.2, lw * 3.4, lw][pass];
        ctx.beginPath(); ctx.moveTo(...P(i)); ctx.lineTo(...P(j)); ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    // 関節：信頼度が高いものは塗りつぶし、低いものは黄色の輪だけ
    for (let k = 0; k < res.kc.length; k++) {
      const [x, y] = P(k), good = res.kc[k] >= KPT_CONF;
      ctx.beginPath(); ctx.arc(x, y, lw * 1.9, 0, Math.PI * 2);
      if (good) {
        ctx.fillStyle = "#0d0f14"; ctx.fill();
        ctx.lineWidth = lw * 0.9; ctx.strokeStyle = SIDE_COLOR[sideOf(meta.keypoints[k])]; ctx.stroke();
      } else {
        ctx.lineWidth = lw * 0.7; ctx.strokeStyle = "#ffb020"; ctx.setLineDash([lw, lw]); ctx.stroke(); ctx.setLineDash([]);
      }
    }
  }
  ctx.restore();
}

function updateAngles(res) {
  const body = $("angles").tBodies[0];
  body.innerHTML = "";
  for (const [name, [a, b, c]] of Object.entries(ANGLES)) {
    const ok = res && [a, b, c].every((n) => res.kc[idx[n]] >= KPT_CONF);
    const deg = ok ? angle(res.xy, idx[a], idx[b], idx[c]) : NaN;
    const tr = body.insertRow();
    tr.insertCell().textContent = t(name);
    const bar = document.createElement("div");
    bar.className = "meter" + (name.startsWith("l_") ? " l" : name.startsWith("r_") ? " r" : "");
    bar.innerHTML = `<i style="width:${Number.isFinite(deg) ? (deg / 180 * 100).toFixed(0) : 0}%"></i>`;
    tr.insertCell().appendChild(bar);
    tr.insertCell().textContent = Number.isFinite(deg) ? `${deg.toFixed(0)}°` : "—";
  }
}

/** 速度と検出の表示。det は検出の状態（"44%"・"none"・"loading"。null なら「—」） */
let lastDet = null;
function setStatus(det = null) {
  lastDet = det;
  const detText = det === "none" ? t("none") : det === "loading" ? t("loadingShort") : det;
  // カメラの枠の見出しの右に、速度と検出の信頼度だけを文字で出す（style.css の .chip）
  const chip = (cls, label, value, on) => `<span class="chip ${cls}${on ? " on" : ""}">${label}<b>${value}</b></span>`;
  $("status").innerHTML = chip("fps", t("speed"), fps ? `${fps.toFixed(1)} fps` : "—", fps > 0)
    + chip("det", t("detect"), detText ?? "—", !!det && det.endsWith("%"));
}

// ---------------------------------------------------------------- 推論ループ

async function inferOnce(el, w, h, smooth) {
  if (!session) {  // モデルの読み込み前
    setStatus("loading");
    return null;
  }
  const pre = preprocess(el, w, h);
  const out = await session.run({ [session.inputNames[0]]: pre.tensor });
  let res = postprocess(out[session.outputNames[0]], pre, parseFloat($("conf").value));
  if (res && smooth) res = { ...res, xy: smoother.filter(res.xy, performance.now() / 1000) };
  if (!res) smoother.reset();
  if (res && depth) res.dz = await inferDepth(el, w, h, res);
  lastResult = res;
  window.__dummy13 = { result: res, backend };  // テスト用
  // アバター表示（avatar.js）へ渡す
  window.dispatchEvent(new CustomEvent("dummy13-pose", {
    detail: res ? { xy: res.xy, kc: res.kc, dz: res.dz, names: meta.keypoints, kptConf: KPT_CONF, width: w, height: h } : null,
  }));
  return res;
}

async function loop() {
  if (!running) return;
  if (!busy && video.readyState >= 2) {
    busy = true;
    const w = video.videoWidth, h = video.videoHeight;
    const res = await inferOnce(video, w, h, $("smooth").checked);
    draw(video, w, h, res);
    updateAngles(res);
    const now = performance.now();
    fps = 0.9 * fps + 0.1 * (1000 / Math.max(now - lastT, 1));
    lastT = now;
    setStatus(res ? `${(res.conf * 100).toFixed(0)}%` : "none");
    busy = false;
  }
  requestAnimationFrame(loop);
}

// ---------------------------------------------------------------- 入力（カメラ・ファイル）

async function listCameras() {
  const sel = $("camera");
  try {
    const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
    sel.innerHTML = devs.map((d, i) => `<option value="${d.deviceId}">${d.label || t("cameraN", { n: i })}</option>`).join("")
      || `<option value="">${t("noCamera")}</option>`;
  } catch (_) {
    sel.innerHTML = `<option value="">${t("cameraUnavailable")}</option>`;
  }
}

async function startCamera() {
  stopSource();
  const id = $("camera").value;
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { deviceId: id ? { exact: id } : undefined, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
  });
  video.srcObject = stream;
  await video.play();
  await listCameras();  // 許可後はカメラ名が取れる
  if (id) $("camera").value = id;
  startLoop();
}

function startLoop() {
  overlay.hidden = true;
  smoother.reset();
  running = true;
  requestAnimationFrame(loop);
}

let still = null;  // 静止画を表示しているときの画像と結果（表示の切り替えで描き直すため）

function stopSource() {
  running = false;
  still = null;
  if (video.srcObject) { video.srcObject.getTracks().forEach((t) => t.stop()); video.srcObject = null; }
  video.removeAttribute("src");
}

async function openFile(file) {
  stopSource();
  const url = URL.createObjectURL(file);
  if (file.type.startsWith("video/")) {
    video.src = url;
    video.loop = true;
    await video.play();
    startLoop();
  } else {
    const img = new Image();
    img.src = url;
    await img.decode();
    overlay.hidden = true;
    const res = await inferOnce(img, img.naturalWidth, img.naturalHeight, false);
    still = { img, res };
    draw(img, img.naturalWidth, img.naturalHeight, res);
    updateAngles(res);
    setStatus(res ? `${(res.conf * 100).toFixed(0)}%` : "none");
  }
}

// ---------------------------------------------------------------- 起動

async function main() {
  $("conf").addEventListener("input", (e) => { $("confVal").textContent = (+e.target.value).toFixed(2); });
  // 静止画のときは描き直しが起きないので、表示の切り替えで描き直す（カメラ・動画は次のフレームで反映される）
  for (const id of ["bones", "mirror"]) {
    $(id).addEventListener("change", () => { if (still) draw(still.img, still.img.naturalWidth, still.img.naturalHeight, still.res); });
  }
  $("start").addEventListener("click", () => startCamera().catch((e) => {
    overlay.hidden = false;
    setOverlay("cameraError", { msg: e.message });
  }));
  $("file").addEventListener("change", (e) => e.target.files[0] && openFile(e.target.files[0]));
  $("modelFile").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (f) await loadModel(new Uint8Array(await f.arrayBuffer())).catch((err) => { overlayMsg = null; overlay.textContent = err.message; });
  });
  // 言語を切り替えたら、スクリプトで書いた表示も今の言語で書き直す
  window.addEventListener("dummy13-lang", () => {
    if (overlayMsg) overlay.textContent = t(...overlayMsg);
    setStatus(lastDet);
    updateAngles(still ? still.res : lastResult);
    const cam = $("camera").value; listCameras().then(() => { if (cam) $("camera").value = cam; });
  });
  await listCameras();
  await loadMeta();
  loadDepth();  // 奥行きモデルは無くても動く（待たない）
  try {
    await loadModel(MODEL_URL);
  } catch (e) {
    // モデルが置かれていない場合は手元の .onnx を選んでもらう
    $("modelPick").hidden = false;
    setOverlay("modelMissing");
  }
}

main();
