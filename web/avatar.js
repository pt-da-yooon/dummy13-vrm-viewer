// 推定した Dummy13 のポーズで 3D アバターを動かす（ブラウザ版の横に表示）。
// VRM を読み込めばそのアバター、無ければ簡単なマネキンを動かす。
// 奥行きは Dummy13 の骨の長さから推定する（画面上で短く見える骨ほど手前／奥に傾いている）。
// 手前か奥かは見た目だけでは決まらないので、体の向きと関節の曲がる向きから決める（推定なので外れることもある）。
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";

// [VRM のボーン名, 根元のキーポイント, 先端のキーポイント, 休止姿勢での向きを測る子ボーン]
// 親 → 子の順に並べる（親の回転を先に決めてから子を決めるため）
const CHAIN = [
  ["hips", "hips", "neck", "head"],
  ["head", "neck", "head", null],
  ["leftUpperArm", "l_shoulder", "l_elbow", "leftLowerArm"],
  ["leftLowerArm", "l_elbow", "l_wrist", "leftHand"],
  ["rightUpperArm", "r_shoulder", "r_elbow", "rightLowerArm"],
  ["rightLowerArm", "r_elbow", "r_wrist", "rightHand"],
  ["leftUpperLeg", "l_hip", "l_knee", "leftLowerLeg"],
  ["leftLowerLeg", "l_knee", "l_ankle", "leftFoot"],
  ["rightUpperLeg", "r_hip", "r_knee", "rightLowerLeg"],
  ["rightLowerLeg", "r_knee", "r_ankle", "rightFoot"],
];
const FOLLOW = 0.5;  // 1 フレームで目標の回転にどれだけ近づけるか（0〜1）

// Dummy13 のキーポイント間の長さ（mm、synth/dummy13_layout.py の T ポーズから計算）
const BONE_MM = {
  "hips-chest": 34.0, "chest-neck": 10.0, "neck-head": 14.5,
  "neck-l_shoulder": 14.32, "l_shoulder-l_elbow": 22.2, "l_elbow-l_wrist": 18.0,
  "neck-r_shoulder": 14.32, "r_shoulder-r_elbow": 22.2, "r_elbow-r_wrist": 18.0,
  "hips-l_hip": 8.0, "l_hip-l_knee": 30.2, "l_knee-l_ankle": 31.0,
  "hips-r_hip": 8.0, "r_hip-r_knee": 30.2, "r_knee-r_ankle": 31.0,
};
// 腰からたどる順番と、奥行きの向きの決め方
//   front: 体の正面側（腕・太ももは前に出すことが多い）  back: 背中側（すねは膝から後ろへ曲がる）
//   none: 決めようがないので奥行き 0（肩・股の横方向）
const TREE = [
  ["hips", "chest", "front"], ["chest", "neck", "front"], ["neck", "head", "front"],
  ["neck", "l_shoulder", "none"], ["l_shoulder", "l_elbow", "front"], ["l_elbow", "l_wrist", "front"],
  ["neck", "r_shoulder", "none"], ["r_shoulder", "r_elbow", "front"], ["r_elbow", "r_wrist", "front"],
  ["hips", "l_hip", "none"], ["l_hip", "l_knee", "front"], ["l_knee", "l_ankle", "back"],
  ["hips", "r_hip", "none"], ["r_hip", "r_knee", "front"], ["r_knee", "r_ankle", "back"],
];
const SCALE_MIN_MM = 15;  // 尺度を測るのに使う骨の最短の長さ
let pxPerMm = 0;  // 画面上の 1mm が何 px か（ゆっくり追従させる）
const opt = { depth: true };

// ---------------------------------------------------------------- 奥行き推定モデル（lifter）
// train/train_lifter.py が書き出す models/lifter.json（全身の 2D の形から各関節の手前／奥を当てる MLP）。
// 無ければ骨の長さと決め打ちの向き（TREE の front/back）だけで奥行きを付ける。
let lifter = null;
const f32 = (b64) => { const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new Float32Array(u.buffer); };
fetch("models/lifter.json").then((r) => (r.ok ? r.json() : null)).then((j) => {
  if (!j) return;
  lifter = { ...j, layers: j.layers.map((l) => ({ ...l, w: f32(l.w), b: f32(l.b) })) };
  window.__lifterLoaded = true;
  console.log("2D→奥行きモデル（lifter）を読み込みました");
}).catch(() => {});

/** lifter で各関節の奥行き（腰から手前が +、mm）を推定する。入力の作り方は train_lifter.py の make_inputs と同じ */
function runLifter(p2, ok) {
  const names = lifter.keypoints, K = names.length, U = lifter.unit_mm;
  const [hx, hy] = p2(lifter.hips);
  let s = 0;
  for (const [a, b, L] of lifter.bones) {
    if (!ok(a) || !ok(b)) continue;
    const [ax, ay] = p2(a), [bx, by] = p2(b);
    s = Math.max(s, Math.hypot(bx - ax, by - ay) / L);
  }
  if (s <= 0) return null;
  let x = new Float32Array(3 * K);
  names.forEach((n, i) => {
    if (!ok(n)) return;
    const [px, py] = p2(n);
    x[2 * i] = (px - hx) / s / U; x[2 * i + 1] = (py - hy) / s / U; x[2 * K + i] = 1;
  });
  lifter.layers.forEach((l, li) => {
    const y = new Float32Array(l.out);
    for (let o = 0; o < l.out; o++) {
      let v = l.b[o];
      const row = o * l.in;
      for (let i = 0; i < l.in; i++) v += l.w[row + i] * x[i];
      y[o] = li < lifter.layers.length - 1 ? Math.max(v, 0) : v;
    }
    x = y;
  });
  return Object.fromEntries(names.map((n, i) => [n, x[i] * U]));
}

const canvas = document.getElementById("avatar");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30, 1, 0.05, 50);
camera.position.set(0, 1.1, 5.2);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 1.0, 0);
controls.update();
scene.add(new THREE.HemisphereLight(0xffffff, 0x666677, 2.2));
const sun = new THREE.DirectionalLight(0xffffff, 2.0);
sun.position.set(1, 2, 3);
scene.add(sun);
// 床の格子。背景を明るくしたときは、細かい線を明るい灰色に、中心の線を濃いライムにする
function makeGrid(light) {
  const g = new THREE.GridHelper(4, 16, light ? 0x7e9c25 : 0xb4d64a, light ? 0xaab3c0 : 0x2a3242);
  g.material.transparent = true;
  g.material.opacity = light ? 0.6 : 0.35;
  return g;
}
let grid = makeGrid(false);
scene.add(grid);

// カメラ画像に重ねる表示（左の画面の上に透明なキャンバスを重ね、画像のピクセル座標で描く）
const ovCanvas = document.getElementById("overlay3d");
const ovRenderer = new THREE.WebGLRenderer({ canvas: ovCanvas, antialias: true, alpha: true });
ovRenderer.setPixelRatio(window.devicePixelRatio);
const ovCamera = new THREE.OrthographicCamera(0, 640, 0, -480, -1e5, 1e5);
const DUMMY_TORSO_MM = 44.0;  // Dummy13 の腰〜首（hips-chest 34 + chest-neck 10）
let lastFrame = null;         // { hips: [x, y], width, height }（画像のピクセル座標）

let rig = null;   // { root, bone(name) → Object3D, rest: name → {dir, q}, update(dt) }
const clock = new THREE.Clock();

// ---------------------------------------------------------------- リグ（VRM／マネキン共通）

function captureRest(boneOf) {
  scene.updateMatrixWorld(true);
  const rest = {};
  for (const [name, , , child] of CHAIN) {
    const node = boneOf(name);
    if (!node) continue;
    const p = node.getWorldPosition(new THREE.Vector3());
    const c = child && boneOf(child);
    const dir = c ? c.getWorldPosition(new THREE.Vector3()).sub(p).normalize() : new THREE.Vector3(0, 1, 0);
    rest[name] = { dir, q: node.getWorldQuaternion(new THREE.Quaternion()) };
  }
  return rest;
}

function setRig(root, boneOf, update = () => {}) {
  if (rig) scene.remove(rig.root);
  scene.add(root);
  rig = { root, bone: boneOf, rest: null, update };
  rig.rest = captureRest(boneOf);
  // 重ね表示で大きさと位置を合わせるための、休止姿勢での腰の位置と胴の長さ
  const hipsB = boneOf("hips"), neckB = boneOf("neck") || boneOf("head");
  if (hipsB && neckB) {
    rig.hipsLocal = hipsB.getWorldPosition(new THREE.Vector3()).sub(root.position);
    rig.torsoM = neckB.getWorldPosition(new THREE.Vector3()).distanceTo(hipsB.getWorldPosition(new THREE.Vector3()));
  }
  const lh = boneOf("leftUpperLeg"), rh = boneOf("rightUpperLeg");
  if (lh && rh) rig.rest.hipsLeft = lh.getWorldPosition(new THREE.Vector3()).sub(rh.getWorldPosition(new THREE.Vector3()));

}

/** 手作りのマネキン（VRM が無いとき用）。VRM 1.0 と同じく +Z 向き・左手側が +X・T ポーズ */
function buildMannequin() {
  const root = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({ color: 0xa3c43f, roughness: 0.6 });
  const joint = new THREE.MeshStandardMaterial({ color: 0x333338, roughness: 0.5 });
  const bones = {};
  const mk = (name, parent, pos) => {
    const o = new THREE.Object3D();
    o.name = name;
    o.position.set(...pos);
    (parent ? bones[parent] : root).add(o);
    bones[name] = o;
    return o;
  };
  const limb = (from, to, r) => {  // from ボーンに、to ボーンまでの太さ r の円柱を付ける
    const a = bones[to].position;
    const len = a.length();
    const g = new THREE.CylinderGeometry(r, r * 0.9, len, 12);
    const m = new THREE.Mesh(g, mat);
    m.position.copy(a).multiplyScalar(0.5);
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), a.clone().normalize());
    bones[from].add(m);
    const s = new THREE.Mesh(new THREE.SphereGeometry(r * 1.15, 16, 12), joint);
    bones[from].add(s);
  };
  mk("hips", null, [0, 0.95, 0]);
  mk("spine", "hips", [0, 0.15, 0]);
  mk("chest", "spine", [0, 0.18, 0]);
  mk("neck", "chest", [0, 0.17, 0]);
  mk("head", "neck", [0, 0.08, 0]);
  mk("headTop", "head", [0, 0.22, 0]);
  for (const [s, x] of [["left", 1], ["right", -1]]) {
    mk(`${s}UpperArm`, "chest", [0.17 * x, 0.13, 0]);
    mk(`${s}LowerArm`, `${s}UpperArm`, [0.27 * x, 0, 0]);
    mk(`${s}Hand`, `${s}LowerArm`, [0.25 * x, 0, 0]);
    mk(`${s}HandTip`, `${s}Hand`, [0.1 * x, 0, 0]);
    mk(`${s}UpperLeg`, "hips", [0.09 * x, -0.05, 0]);
    mk(`${s}LowerLeg`, `${s}UpperLeg`, [0, -0.42, 0]);
    mk(`${s}Foot`, `${s}LowerLeg`, [0, -0.42, 0]);
    mk(`${s}Toe`, `${s}Foot`, [0, -0.04, 0.12]);
  }
  limb("hips", "spine", 0.1); limb("spine", "chest", 0.11); limb("chest", "neck", 0.12);
  limb("neck", "head", 0.04); limb("head", "headTop", 0.1);
  for (const s of ["left", "right"]) {
    limb(`${s}UpperArm`, `${s}LowerArm`, 0.045); limb(`${s}LowerArm`, `${s}Hand`, 0.04);
    limb(`${s}Hand`, `${s}HandTip`, 0.035);
    limb(`${s}UpperLeg`, `${s}LowerLeg`, 0.065); limb(`${s}LowerLeg`, `${s}Foot`, 0.05);
    limb(`${s}Foot`, `${s}Toe`, 0.04);
  }
  setRig(root, (n) => bones[n]);
}

async function loadVRM(url) {
  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));
  const gltf = await loader.loadAsync(url);
  const vrm = gltf.userData.vrm;
  if (!vrm) throw new Error(window.i18n ? window.i18n.t("notVrm") : "VRM ではありません");
  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  VRMUtils.rotateVRM0(vrm);  // VRM 0.x は -Z 向きなので +Z 向きにそろえる
  vrm.scene.traverse((o) => { o.frustumCulled = false; });
  setRig(vrm.scene, (n) => vrm.humanoid.getNormalizedBoneNode(n), (dt) => vrm.update(dt));
  window.__avatar = { kind: "vrm", name: vrm.meta?.name || vrm.meta?.title || "" };
}

// ---------------------------------------------------------------- ポーズの反映

const _v = new THREE.Vector3(), _q = new THREE.Quaternion(), _p = new THREE.Quaternion();
const _m1 = new THREE.Matrix4(), _m2 = new THREE.Matrix4();

/** 2D キーポイントから 3D の位置（mm、x 右・y 上・z 手前）を推定する */
function lift({ xy, kc, names, kptConf, dz }) {
  const id = Object.fromEntries(names.map((n, i) => [n, i]));
  const ok = (n) => id[n] !== undefined && kc[id[n]] >= kptConf;
  const p2 = (n) => [xy[2 * id[n]], -xy[2 * id[n] + 1]];
  // 尺度：骨は画面上では本来の長さ以下にしか見えないので、「見た目の長さ ÷ 本来の長さ」の最大値を使う。
  // 短い骨（首〜頭・肩・股など 15mm 未満）は点が数 px ずれるだけで比が大きく狂い、尺度が大きすぎると
  // ほかの骨がすべて「短く見える＝手前／奥に傾いている」ことになって頭や胴が前後に倒れるので、長い骨だけで測る
  let s = 0;
  for (const [a, b] of TREE) {
    if (!ok(a) || !ok(b) || BONE_MM[`${a}-${b}`] < SCALE_MIN_MM) continue;
    const [ax, ay] = p2(a), [bx, by] = p2(b);
    s = Math.max(s, Math.hypot(bx - ax, by - ay) / BONE_MM[`${a}-${b}`]);
  }
  if (s <= 0) return null;
  pxPerMm = pxPerMm ? Math.max(s, 0.9 * pxPerMm + 0.1 * s) : s;
  // 体の向き：Dummy13 の左が画面の右にあればカメラの方を向いている（正面 = +z）
  let lr = 0;
  if (ok("l_shoulder") && ok("r_shoulder")) lr += p2("l_shoulder")[0] - p2("r_shoulder")[0];
  if (ok("l_hip") && ok("r_hip")) lr += p2("l_hip")[0] - p2("r_hip")[0];
  const facing = lr >= 0 ? 1 : -1;
  const P = {};
  if (!ok("hips")) return null;
  // 手前／奥の向き：画像からの奥行きモデル（app.js）＞ 2D からの lifter ＞ 決め打ち の順に使う
  const dzLift = !opt.depth ? null : dz || (lifter ? runLifter(p2, ok) : null);
  const [hx, hy] = p2("hips");
  P.hips = new THREE.Vector3(0, 0, 0);
  for (const [a, b, rule] of TREE) {
    if (!P[a] || !ok(b)) continue;
    const [ax, ay] = p2(a), [bx, by] = p2(b);
    const L = BONE_MM[`${a}-${b}`] * pxPerMm, l = Math.hypot(bx - ax, by - ay);
    // 奥行きの大きさは「本来の骨の長さ」と「画面上の長さ」の差から決める
    let dz = opt.depth ? Math.sqrt(Math.max(0, L * L - l * l)) / pxPerMm : 0;
    if (dzLift) {
      // 手前／奥の向きはモデルに任せる（迷っているときは 0 に寄せてパタパタしないようにする）
      dz *= Math.tanh((dzLift[b] - dzLift[a]) / (0.3 * BONE_MM[`${a}-${b}`]));
    } else {
      dz = rule === "none" ? 0 : (rule === "back" ? -dz : dz) * facing;
    }
    P[b] = new THREE.Vector3((bx - hx) / pxPerMm, (by - hy) / pxPerMm, P[a].z + dz);
  }
  return P;
}

/** 2 本のベクトル（上・左）で決まる回転 */
function frameQuat(up, left, out) {
  const y = up.clone().normalize();
  const x = left.clone().sub(y.clone().multiplyScalar(left.dot(y)));  // 上と直交させる
  if (x.lengthSq() < 1e-6) return null;
  x.normalize();
  const z = new THREE.Vector3().crossVectors(x, y);
  return out.setFromRotationMatrix(_m1.makeBasis(x, y, z));
}

function setBoneWorld(node, worldQ) {
  node.parent.updateWorldMatrix(true, false);
  node.parent.getWorldQuaternion(_p).invert();
  node.quaternion.slerp(_p.multiply(worldQ), FOLLOW);
  node.updateMatrixWorld(true);
}

/** 胴（腰〜首）の、休止姿勢からの回転。「上」（腰→首）と「左」（右→左の腰・肩）の 2 方向で決める */
function bodyDelta(P) {
  const rest = rig.rest.hips;
  if (!rest || !rig.rest.hipsLeft || !P.hips || !P.neck || !P.l_hip || !P.r_hip) return null;
  const up = new THREE.Vector3().subVectors(P.neck, P.hips);
  const left = new THREE.Vector3().subVectors(P.l_hip, P.r_hip);
  if (P.l_shoulder && P.r_shoulder) left.add(new THREE.Vector3().subVectors(P.l_shoulder, P.r_shoulder));
  const qt = frameQuat(up, left, new THREE.Quaternion());
  const qr = frameQuat(rest.dir, rig.rest.hipsLeft, new THREE.Quaternion());
  return qt && qr ? qt.multiply(qr.invert()) : null;
}

function applyPose(detail) {
  if (!rig) return;
  const P = lift(detail);
  if (!P) return;
  const hi = detail.names.indexOf("hips");
  lastFrame = { hips: [detail.xy[2 * hi], detail.xy[2 * hi + 1]], width: detail.width, height: detail.height };
  const D = bodyDelta(P);
  for (const [name, a, b] of CHAIN) {
    const node = rig.bone(name), rest = rig.rest[name];
    if (!node || !rest) continue;
    if (!P[a] || !P[b]) continue;
    _v.subVectors(P[b], P[a]);
    if (_v.lengthSq() < 1e-6) continue;
    _v.normalize();
    if (name === "hips" && D) {
      // 腰は「上」と「左」の 2 方向で向きを決める（背中を向けたときに体ごと回るように）
      setBoneWorld(node, D.clone().multiply(rest.q));
      continue;
    }
    // 目標のワールド回転 = 休止姿勢の向き → 目標の向き への回転 × 休止姿勢のワールド回転
    _q.setFromUnitVectors(rest.dir, _v).multiply(rest.q);
    setBoneWorld(node, _q);
  }
}

window.addEventListener("dummy13-pose", (e) => { if (e.detail) applyPose(e.detail); });

// ---------------------------------------------------------------- 画面

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.floor(w * renderer.getPixelRatio()) || canvas.height !== Math.floor(h * renderer.getPixelRatio())) {
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}

/** アバターを、画像の上で Dummy13 と同じ位置・大きさになるように一時的に動かして描く */
function renderOverlay() {
  const on = document.getElementById("overlayMode").checked;
  ovCanvas.hidden = !on;
  if (!on || !rig || !lastFrame || !pxPerMm || !rig.torsoM || !lastFrame.width) {
    if (on) ovRenderer.clear();
    return;
  }
  ovCanvas.classList.toggle("mirror", document.getElementById("mirror").checked);
  // 重ねる先（カメラ画像のキャンバス）と同じ表示サイズにする
  const img = document.getElementById("canvas");
  const w = img.clientWidth, h = img.clientHeight;
  if (ovCanvas.style.height !== h + "px") { ovCanvas.style.width = w + "px"; ovCanvas.style.height = h + "px"; }
  if (ovCanvas.width !== Math.floor(w * ovRenderer.getPixelRatio()) || ovCanvas.height !== Math.floor(h * ovRenderer.getPixelRatio())) ovRenderer.setSize(w, h, false);
  ovCamera.left = 0; ovCamera.right = lastFrame.width; ovCamera.top = 0; ovCamera.bottom = -lastFrame.height;
  ovCamera.updateProjectionMatrix();
  const root = rig.root;
  const pos0 = root.position.clone(), scl0 = root.scale.clone();
  const k = (DUMMY_TORSO_MM * pxPerMm) / rig.torsoM;  // 1m を何 px にするか
  root.scale.setScalar(k);
  root.position.set(lastFrame.hips[0], -lastFrame.hips[1], 0).sub(rig.hipsLocal.clone().multiplyScalar(k));
  grid.visible = false;
  root.updateMatrixWorld(true);
  ovRenderer.render(scene, ovCamera);
  root.position.copy(pos0); root.scale.copy(scl0); grid.visible = true;
  root.updateMatrixWorld(true);
}

function tick() {
  resize();
  const dt = clock.getDelta();
  if (rig) rig.update(dt);
  renderer.render(scene, camera);
  renderOverlay();
  requestAnimationFrame(tick);
}

document.getElementById("vrmFile").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const url = URL.createObjectURL(f);
  const label = document.getElementById("avatarName");
  try {
    await loadVRM(url);
    label.removeAttribute("data-i18n");  // 名前は言語で変えない
    label.textContent = window.__avatar.name || f.name;
  } catch (err) {
    console.error(err);
    label.removeAttribute("data-i18n");
    label.textContent = window.i18n ? window.i18n.t("vrmFailed", { msg: err.message }) : `読み込めませんでした: ${err.message}`;
  } finally {
    URL.revokeObjectURL(url);
  }
});
document.getElementById("useDepth").addEventListener("change", (e) => { opt.depth = e.target.checked; });
// 3D 表示の背景だけを明るくする（画面全体の色は変えない）。選んだものはこのブラウザに覚えておく
const BG_KEY = "dummy13-avatar-bg";
function setLightBg(light) {
  canvas.parentElement.classList.toggle("light-bg", light);
  const visible = grid.visible;
  scene.remove(grid); grid.geometry.dispose(); grid.material.dispose();
  grid = makeGrid(light); grid.visible = visible; scene.add(grid);
}
{
  let saved = null;
  try { saved = localStorage.getItem(BG_KEY); } catch { /* 保存できない環境では毎回暗い背景 */ }
  const sw = document.getElementById("lightBg");
  if (sw) {
    sw.checked = saved === "light";
    sw.addEventListener("change", () => {
      setLightBg(sw.checked);
      try { localStorage.setItem(BG_KEY, sw.checked ? "light" : "dark"); } catch { /* 切り替えはできる */ }
    });
  }
  if (saved === "light") setLightBg(true);
}
document.getElementById("resetView").addEventListener("click", () => {
  camera.position.set(0, 1.1, 5.2);
  controls.target.set(0, 1.0, 0);
  controls.update();
});

buildMannequin();
window.__avatar = { kind: "mannequin" };
tick();
