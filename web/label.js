// Dummy13 点付けツール：実物の写真に関節の点を付けて YOLO-pose 形式のラベルを保存する。
// 写真は <選んだフォルダ>/images/、ラベルは <選んだフォルダ>/labels/<同じ名前>.txt（train/train.py --real で学習に混ぜる）。
// 今のモデルで下書きし、ドラッグで直す。フォルダの読み書きは File System Access API（Chrome / Edge）。
import { detect, loadMeta, loadSession } from "./pose.js";

const NAMES_JA = {
  head: "頭", neck: "首", chest: "胸", hips: "腰",
  l_shoulder: "左肩", l_elbow: "左ひじ", l_wrist: "左手首", r_shoulder: "右肩", r_elbow: "右ひじ", r_wrist: "右手首",
  l_hip: "左股", l_knee: "左ひざ", l_ankle: "左足首", r_hip: "右股", r_knee: "右ひざ", r_ankle: "右足首",
};
const VIS_JA = { 2: "見えている", 1: "隠れている", 0: "写っていない" };
const IMAGE_EXT = /\.(jpe?g|png|webp|bmp)$/i;
const KPT_CONF = 0.5;  // 下書きでこれ未満の点は「要確認」にする

const $ = (id) => document.getElementById(id);
const canvas = $("canvas"), ctx = canvas.getContext("2d");
const video = $("video"), overlay = $("overlay");

let meta = null, session = null;
let dir = null, imagesDir = null, labelsDir = null;
let files = [];            // [{ name, handle }]
const labeled = new Set(); // ラベルのある写真の名前
let cur = null;            // { index, name, img, w, h, kps: [[x, y, v]], box: [x0, y0, x1, y1], check: [bool], dirty, from }
let sel = 0;               // 選んでいる点
let view = { s: 1, ox: 0, oy: 0 };
let drag = null;           // { kind: "kp" | "corner" | "pan", ... }
const undoStack = [];      // 元に戻す用（kps と box の写し）
let stream = null;

const color = (n) => (n.startsWith("l_") ? "#3b82f6" : n.startsWith("r_") ? "#ef4444" : "#22c55e");
const stem = (name) => name.replace(/\.[^.]+$/, "");

// ---------------------------------------------------------------- フォルダ

async function openDir() {
  if (!window.showDirectoryPicker) {
    alert("このブラウザはフォルダへの保存に対応していません。Chrome か Edge で開いてください。");
    return;
  }
  dir = await window.showDirectoryPicker({ mode: "readwrite" });
  imagesDir = await dir.getDirectoryHandle("images", { create: true });
  labelsDir = await dir.getDirectoryHandle("labels", { create: true });
  files = [];
  labeled.clear();
  for await (const [name, h] of imagesDir.entries()) if (h.kind === "file" && IMAGE_EXT.test(name)) files.push({ name, handle: h });
  for await (const [name, h] of labelsDir.entries()) if (h.kind === "file" && name.endsWith(".txt")) labeled.add(name.slice(0, -4));
  files.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  $("dirName").textContent = `${dir.name}/images（${files.length} 枚）`;
  if (!files.length) {
    overlay.hidden = false;
    overlay.textContent = `${dir.name}/images/ に写真がありません。写真を入れるか、「カメラで撮る」で撮影してください。`;
    updateSide();
    return;
  }
  // まだラベルの無い最初の写真から始める
  const first = files.findIndex((f) => !labeled.has(stem(f.name)));
  await openImage(first >= 0 ? first : 0);
}

async function readLabel(name) {
  try {
    const f = await (await labelsDir.getFileHandle(`${stem(name)}.txt`)).getFile();
    return await f.text();
  } catch (_) {
    return null;
  }
}

/** YOLO-pose のラベル（1 行目だけ使う）→ { box, kps } */
function parseLabel(text, w, h) {
  const v = text.trim().split(/\s+/).map(Number);
  if (v.length < 5 + 3 * meta.keypoints.length) return null;
  const [cx, cy, bw, bh] = v.slice(1, 5);
  const box = [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h];
  const kps = meta.keypoints.map((_, k) => [v[5 + 3 * k] * w, v[6 + 3 * k] * h, v[7 + 3 * k]]);
  return { box, kps };
}

function formatLabel({ box, kps, w, h }) {
  const c = (x) => Math.min(Math.max(x, 0), 1).toFixed(6);
  const [x0, y0, x1, y1] = box;
  const head = `0 ${c((x0 + x1) / 2 / w)} ${c((y0 + y1) / 2 / h)} ${c((x1 - x0) / w)} ${c((y1 - y0) / h)}`;
  const pts = kps.map(([x, y, v]) => (v > 0 ? `${c(x / w)} ${c(y / h)} ${v}` : "0 0 0"));
  return `${head} ${pts.join(" ")}\n`;
}

async function save() {
  if (!cur || !labelsDir) return;
  const fh = await labelsDir.getFileHandle(`${stem(cur.name)}.txt`, { create: true });
  const w = await fh.createWritable();
  await w.write(formatLabel(cur));
  await w.close();
  labeled.add(stem(cur.name));
  cur.dirty = false;
  cur.from = "保存したラベルを表示中";
  cur.check.fill(false);
  updateSide();
  draw();
}

async function skip() {
  if (!cur) return;
  try { await labelsDir.removeEntry(`${stem(cur.name)}.txt`); } catch (_) { /* もともと無い */ }
  labeled.delete(stem(cur.name));
  cur.dirty = false;
  await go(1);
}

// ---------------------------------------------------------------- 写真を開く・下書き

async function openImage(index) {
  index = Math.max(0, Math.min(files.length - 1, index));
  const f = files[index];
  const img = await createImageBitmap(await f.handle.getFile());
  cur = { index, name: f.name, img, w: img.width, h: img.height, kps: null, box: null, check: [], dirty: false, from: "" };
  undoStack.length = 0;
  const text = await readLabel(f.name);
  const parsed = text && parseLabel(text, cur.w, cur.h);
  if (parsed) {
    Object.assign(cur, parsed, { from: "保存したラベルを表示中" });
    cur.check = cur.kps.map(() => false);
  } else {
    await redetect(false);
  }
  overlay.hidden = true;
  fit();
  updateSide();
  draw();
}

/** 今のモデルで下書きする。見つからなければ画面の中央に立ちポーズの型を置く */
async function redetect(markDirty = true) {
  if (!cur) return;
  pushHistory();
  const res = session ? await detect(session, meta, cur.img, cur.w, cur.h, 0.1) : null;
  if (res) {
    cur.box = [...res.box];
    cur.kps = meta.keypoints.map((_, k) => [res.xy[2 * k], res.xy[2 * k + 1], 2]);
    cur.check = meta.keypoints.map((_, k) => res.kc[k] < KPT_CONF);
    cur.from = `下書き（検出の信頼度 ${(res.conf * 100).toFixed(0)}%）`;
  } else {
    const s = Math.min(cur.w, cur.h) * 0.8;
    placeTemplate([cur.w / 2 - s / 2, cur.h / 2 - s / 2, cur.w / 2 + s / 2, cur.h / 2 + s / 2]);
    boxFromPoints();
    cur.from = "見つからなかったので型を置きました";
  }
  cur.dirty = markDirty;
  updateSide();
  draw();
}

/** 枠の中に、こちらを向いて立つ型を置く（下書きが大きく外れたとき用） */
function placeTemplate([x0, y0, x1, y1]) {
  const T = { head: [0, -0.42], neck: [0, -0.3], chest: [0, -0.18], hips: [0, 0], l_shoulder: [0.1, -0.28],
    l_elbow: [0.14, -0.12], l_wrist: [0.16, 0.02], r_shoulder: [-0.1, -0.28], r_elbow: [-0.14, -0.12],
    r_wrist: [-0.16, 0.02], l_hip: [0.05, 0.02], l_knee: [0.06, 0.22], l_ankle: [0.06, 0.42], r_hip: [-0.05, 0.02],
    r_knee: [-0.06, 0.22], r_ankle: [-0.06, 0.42] };
  const s = (y1 - y0) / 0.95, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  cur.kps = meta.keypoints.map((n) => [cx + (T[n]?.[0] ?? 0) * s, cy + (T[n]?.[1] ?? 0) * s, 2]);
  cur.check = meta.keypoints.map(() => true);
}

function boxFromPoints() {
  const pts = cur.kps.filter((p) => p[2] > 0);
  if (!pts.length) return;
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const m = 0.12 * Math.max(x1 - x0, y1 - y0);  // 関節は体の内側なので少し広げる
  cur.box = [Math.max(0, x0 - m), Math.max(0, y0 - m), Math.min(cur.w, x1 + m), Math.min(cur.h, y1 + m)];
}

async function go(step) {
  if (!files.length) return;
  if (cur?.dirty) await save();
  const i = (cur ? cur.index : 0) + step;
  if (i < 0 || i >= files.length) return;
  await openImage(i);
}

function pushHistory() {
  if (!cur?.kps) return;
  undoStack.push({ kps: cur.kps.map((p) => [...p]), box: [...cur.box] });
  if (undoStack.length > 50) undoStack.shift();
}

function undo() {
  const h = undoStack.pop();
  if (!h || !cur) return;
  cur.kps = h.kps; cur.box = h.box; cur.dirty = true;
  updateSide(); draw();
}

function swapLeftRight() {
  if (!cur) return;
  pushHistory();
  const id = Object.fromEntries(meta.keypoints.map((n, i) => [n, i]));
  const kps = cur.kps.map((p) => [...p]);
  meta.keypoints.forEach((n, i) => {
    const m = n.startsWith("l_") ? `r_${n.slice(2)}` : n.startsWith("r_") ? `l_${n.slice(2)}` : n;
    kps[i] = [...cur.kps[id[m]]];
  });
  cur.kps = kps;
  cur.dirty = true;
  updateSide(); draw();
}

function cycleVis() {
  if (!cur) return;
  pushHistory();
  const p = cur.kps[sel];
  p[2] = p[2] === 2 ? 1 : p[2] === 1 ? 0 : 2;
  cur.check[sel] = false;
  cur.dirty = true;
  updateSide(); draw();
}

// ---------------------------------------------------------------- 表示

function fit() {
  const r = canvas.getBoundingClientRect();
  if (!cur || !r.width) return;
  const s = Math.min(r.width / cur.w, r.height / cur.h);
  view = { s, ox: (r.width - cur.w * s) / 2, oy: (r.height - cur.h * s) / 2 };
}

const toScreen = ([x, y]) => [x * view.s + view.ox, y * view.s + view.oy];
const toImage = (sx, sy) => [(sx - view.ox) / view.s, (sy - view.oy) / view.s];

function draw() {
  const r = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(r.width * dpr) || canvas.height !== Math.round(r.height * dpr)) {
    canvas.width = Math.round(r.width * dpr); canvas.height = Math.round(r.height * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  if (!cur?.kps) return;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(cur.img, view.ox, view.oy, cur.w * view.s, cur.h * view.s);
  // 検出枠
  const [bx0, by0] = toScreen(cur.box), [bx1, by1] = toScreen([cur.box[2], cur.box[3]]);
  ctx.strokeStyle = "rgba(180,214,74,0.9)"; ctx.lineWidth = 1.5; ctx.setLineDash([6, 4]);
  ctx.strokeRect(bx0, by0, bx1 - bx0, by1 - by0); ctx.setLineDash([]);
  ctx.fillStyle = "rgba(180,214,74,0.9)";
  for (const [x, y] of [[bx0, by0], [bx1, by0], [bx0, by1], [bx1, by1]]) ctx.fillRect(x - 4, y - 4, 8, 8);
  // 骨格
  const id = Object.fromEntries(meta.keypoints.map((n, i) => [n, i]));
  ctx.lineWidth = 2;
  for (const [a, b] of meta.skeleton) {
    const pa = cur.kps[id[a]], pb = cur.kps[id[b]];
    if (!pa[2] || !pb[2]) continue;
    const [ax, ay] = toScreen(pa), [bx, by] = toScreen(pb);
    ctx.strokeStyle = b.startsWith("l_") ? "rgba(59,130,246,.9)" : b.startsWith("r_") ? "rgba(239,68,68,.9)" : "rgba(255,255,255,.9)";
    ctx.setLineDash(pa[2] === 1 || pb[2] === 1 ? [4, 3] : []);
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
  }
  ctx.setLineDash([]);
  // 点（選んでいる点は大きく、隠れている点は中抜き、要確認はオレンジの輪）
  meta.keypoints.forEach((n, k) => {
    const p = cur.kps[k];
    if (!p[2]) return;
    const [x, y] = toScreen(p), rad = k === sel ? 7 : 5;
    ctx.beginPath(); ctx.arc(x, y, rad, 0, Math.PI * 2);
    ctx.fillStyle = color(n);
    if (p[2] === 2) ctx.fill();
    ctx.lineWidth = p[2] === 1 ? 2.5 : 1.5;
    ctx.strokeStyle = p[2] === 1 ? color(n) : "#fff";
    ctx.stroke();
    if (cur.check[k]) { ctx.beginPath(); ctx.arc(x, y, rad + 4, 0, Math.PI * 2); ctx.strokeStyle = "#ff9f1c"; ctx.lineWidth = 2; ctx.stroke(); }
    if (k === sel) {
      ctx.font = "bold 13px system-ui, sans-serif";
      ctx.lineWidth = 3; ctx.strokeStyle = "rgba(0,0,0,.7)";
      ctx.strokeText(NAMES_JA[n] || n, x + 10, y - 10);
      ctx.fillStyle = "#fff"; ctx.fillText(NAMES_JA[n] || n, x + 10, y - 10);
    }
  });
}

function updateSide() {
  $("pos").textContent = files.length ? `${(cur?.index ?? 0) + 1} / ${files.length}` : "0 / 0";
  const body = $("kps").tBodies[0];
  body.innerHTML = "";
  if (cur?.kps) {
    meta.keypoints.forEach((n, k) => {
      const tr = body.insertRow();
      if (k === sel) tr.className = "sel";
      const a = tr.insertCell();
      a.innerHTML = `<span class="sw" style="background:${color(n)}"></span>${NAMES_JA[n] || n}${cur.check[k] ? " ⚠" : ""}`;
      tr.insertCell().textContent = VIS_JA[cur.kps[k][2]];
      tr.onclick = () => { sel = k; updateSide(); draw(); };
    });
  }
  const checks = cur?.check.filter(Boolean).length ?? 0;
  $("state").textContent = cur
    ? `${cur.name}\n${labeled.has(stem(cur.name)) ? "保存済み" : "未保存"}${cur.dirty ? "（変更あり）" : ""} ／ ${cur.from}` +
      `${checks ? `\n⚠ 要確認の点 ${checks} 個（オレンジの輪）` : ""}\n点付け済み ${labeled.size} / ${files.length} 枚`
    : "";
}

// ---------------------------------------------------------------- マウス・キー

function hitPoint(sx, sy) {
  let best = -1, bd = 12;
  cur.kps.forEach((p, k) => {
    if (!p[2]) return;
    const [x, y] = toScreen(p), d = Math.hypot(x - sx, y - sy);
    if (d < bd || (d === bd && k === sel)) { bd = d; best = k; }
  });
  return best;
}

function hitCorner(sx, sy) {
  const [x0, y0] = toScreen(cur.box), [x1, y1] = toScreen([cur.box[2], cur.box[3]]);
  const cs = [[x0, y0, 0, 1], [x1, y0, 2, 1], [x0, y1, 0, 3], [x1, y1, 2, 3]];
  return cs.find(([x, y]) => Math.hypot(x - sx, y - sy) < 9) || null;
}

canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("pointerdown", (e) => {
  if (!cur?.kps) return;
  const r = canvas.getBoundingClientRect(), sx = e.clientX - r.left, sy = e.clientY - r.top;
  canvas.setPointerCapture(e.pointerId);
  if (e.button === 2 || e.button === 1) { drag = { kind: "pan", sx, sy, ox: view.ox, oy: view.oy }; return; }
  const k = hitPoint(sx, sy);
  if (k >= 0) {
    pushHistory();
    sel = k; drag = { kind: "kp", k };
    updateSide(); draw();
    return;
  }
  const c = hitCorner(sx, sy);
  if (c) { pushHistory(); drag = { kind: "corner", ix: c[2], iy: c[3] }; }
});
canvas.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const r = canvas.getBoundingClientRect(), sx = e.clientX - r.left, sy = e.clientY - r.top;
  if (drag.kind === "pan") {
    view.ox = drag.ox + sx - drag.sx; view.oy = drag.oy + sy - drag.sy;
  } else {
    const [x, y] = toImage(sx, sy);
    const cx = Math.min(Math.max(x, 0), cur.w), cy = Math.min(Math.max(y, 0), cur.h);
    if (drag.kind === "kp") {
      const p = cur.kps[drag.k];
      p[0] = cx; p[1] = cy;
      cur.check[drag.k] = false;
    } else {
      cur.box[drag.ix] = cx; cur.box[drag.iy] = cy;
    }
    cur.dirty = true;
  }
  draw();
});
canvas.addEventListener("pointerup", () => {
  if (drag?.kind === "corner") {  // 角を入れ替えて引っ張っても x0 < x1, y0 < y1 にそろえる
    const [a, b, c, d] = cur.box;
    cur.box = [Math.min(a, c), Math.min(b, d), Math.max(a, c), Math.max(b, d)];
  }
  if (drag && drag.kind !== "pan") updateSide();
  drag = null;
});
canvas.addEventListener("dblclick", (e) => {
  if (!cur?.kps) return;
  const r = canvas.getBoundingClientRect();
  pushHistory();
  const [x, y] = toImage(e.clientX - r.left, e.clientY - r.top);
  const p = cur.kps[sel];
  p[0] = Math.min(Math.max(x, 0), cur.w); p[1] = Math.min(Math.max(y, 0), cur.h);
  if (!p[2]) p[2] = 2;
  cur.check[sel] = false; cur.dirty = true;
  updateSide(); draw();
});
canvas.addEventListener("wheel", (e) => {
  if (!cur) return;
  e.preventDefault();
  const r = canvas.getBoundingClientRect(), sx = e.clientX - r.left, sy = e.clientY - r.top;
  const [x, y] = toImage(sx, sy);
  view.s *= Math.exp(-e.deltaY * 0.0015);
  view.ox = sx - x * view.s; view.oy = sy - y * view.s;
  draw();
}, { passive: false });

window.addEventListener("keydown", async (e) => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  if (stream && (e.key === "c" || e.key === "C")) { e.preventDefault(); await capture(); return; }
  if (!cur) return;
  const K = meta.keypoints.length;
  if (e.key === "ArrowRight") { e.preventDefault(); await go(1); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); await go(-1); }
  else if (e.key === "Enter") { e.preventDefault(); await save(); await go(1); }
  else if (e.key === "Tab") { e.preventDefault(); sel = (sel + (e.shiftKey ? K - 1 : 1)) % K; updateSide(); draw(); }
  else if (e.key === "s" || e.key === "S") { e.preventDefault(); await save(); }
  else if (e.key === "h" || e.key === "H") cycleVis();
  else if (e.key === "x" || e.key === "X") swapLeftRight();
  else if (e.key === "r" || e.key === "R") await redetect();
  else if (e.key === "b" || e.key === "B") { pushHistory(); boxFromPoints(); cur.dirty = true; updateSide(); draw(); }
  else if (e.key === "f" || e.key === "F") { fit(); draw(); }
  else if (e.key === "t" || e.key === "T") { pushHistory(); placeTemplate(cur.box); cur.dirty = true; updateSide(); draw(); }
  else if ((e.key === "z" || e.key === "Z") && (e.ctrlKey || e.metaKey)) { e.preventDefault(); undo(); }
});
window.addEventListener("resize", () => { fit(); draw(); });

// ---------------------------------------------------------------- カメラで撮る

async function listCameras() {
  try {
    const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
    const sel = $("camera"), keep = sel.value;
    sel.innerHTML = devs.map((d, i) => `<option value="${d.deviceId}">${d.label || `カメラ ${i + 1}`}</option>`).join("");
    if (keep) sel.value = keep;
  } catch (_) { /* カメラが無い */ }
}

async function toggleCamera() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
    video.hidden = true; $("capture").hidden = true; $("camToggle").textContent = "カメラで撮る";
    // 撮った写真の点付けへ（まだラベルの無い最初の写真）
    const first = files.findIndex((f) => !labeled.has(stem(f.name)));
    if (first >= 0) await openImage(first);
    return;
  }
  if (!imagesDir) { alert("先に「フォルダを開く」で保存先を選んでください"); return; }
  const id = $("camera").value;
  stream = await navigator.mediaDevices.getUserMedia({
    video: { ...(id ? { deviceId: { exact: id } } : {}), width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
  });
  video.srcObject = stream;
  await video.play();
  video.hidden = false; overlay.hidden = true; $("capture").hidden = false; $("camToggle").textContent = "撮影をやめる";
  await listCameras();
}

let shots = 0;
async function capture() {
  if (!stream || !video.videoWidth) return;
  const c = document.createElement("canvas");
  c.width = video.videoWidth; c.height = video.videoHeight;
  c.getContext("2d").drawImage(video, 0, 0);
  const blob = await new Promise((r) => c.toBlob(r, "image/jpeg", 0.95));
  const t = new Date(), p = (n) => String(n).padStart(2, "0");
  const name = `cap_${t.getFullYear()}${p(t.getMonth() + 1)}${p(t.getDate())}_${p(t.getHours())}${p(t.getMinutes())}${p(t.getSeconds())}_${shots++}.jpg`;
  const fh = await imagesDir.getFileHandle(name, { create: true });
  const w = await fh.createWritable();
  await w.write(blob);
  await w.close();
  files.push({ name, handle: fh });
  $("dirName").textContent = `${dir.name}/images（${files.length} 枚）`;
  $("capture").textContent = `撮影（C）… ${shots} 枚撮りました`;
  updateSide();
}

// ---------------------------------------------------------------- 起動

$("openDir").onclick = () => openDir().catch((e) => { if (e.name !== "AbortError") alert(e.message); });
$("prev").onclick = () => go(-1);
$("next").onclick = () => go(1);
$("save").onclick = () => save();
$("redetect").onclick = () => redetect();
$("swap").onclick = () => swapLeftRight();
$("skip").onclick = () => skip();
$("camToggle").onclick = () => toggleCamera().catch((e) => alert(`カメラを開けません: ${e.message}`));
$("capture").onclick = () => capture();

meta = await loadMeta();
listCameras();
try {
  ({ session } = await loadSession());
} catch (e) {
  $("dirName").textContent = "モデル（models/dummy13-pose.onnx）が無いので下書きなしで点付けします";
}
window.__label = { get cur() { return cur; }, get files() { return files; }, labeled, formatLabel, openDir, go, save };  // テスト用
