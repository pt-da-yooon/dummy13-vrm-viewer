// YOLO-pose（ONNX）の読み込み・前処理・後処理（点付けツール label.js から使う）。
// 中身は app.js の同名の処理と同じ。ort（ONNX Runtime Web）は <script> で読み込んでおく。

export const ORT_VERSION = "1.30.0";

export const DEFAULT_META = {
  imgsz: 640,
  keypoints: ["head", "neck", "chest", "hips", "l_shoulder", "l_elbow", "l_wrist", "r_shoulder", "r_elbow",
              "r_wrist", "l_hip", "l_knee", "l_ankle", "r_hip", "r_knee", "r_ankle"],
  skeleton: [["head", "neck"], ["neck", "chest"], ["chest", "hips"], ["neck", "l_shoulder"],
             ["l_shoulder", "l_elbow"], ["l_elbow", "l_wrist"], ["neck", "r_shoulder"], ["r_shoulder", "r_elbow"],
             ["r_elbow", "r_wrist"], ["hips", "l_hip"], ["l_hip", "l_knee"], ["l_knee", "l_ankle"],
             ["hips", "r_hip"], ["r_hip", "r_knee"], ["r_knee", "r_ankle"]],
};

export async function loadMeta(url = "models/model.json") {
  try {
    const r = await fetch(url);
    if (r.ok) return { ...DEFAULT_META, ...(await r.json()) };
  } catch (_) { /* 既定値を使う */ }
  return DEFAULT_META;
}

/** WebGPU → WebAssembly の順に試して読み込む。戻り値 { session, backend } */
export async function loadSession(url = "models/dummy13-pose.onnx") {
  ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  for (const ep of navigator.gpu ? ["webgpu", "wasm"] : ["wasm"]) {
    try {
      const session = await ort.InferenceSession.create(url, { executionProviders: [ep], graphOptimizationLevel: "all" });
      return { session, backend: ep === "webgpu" ? "WebGPU（GPU）" : "WebAssembly（CPU）" };
    } catch (e) {
      console.warn(`${ep} で読み込めませんでした`, e);
    }
  }
  throw new Error("モデルを読み込めませんでした");
}

const prep = document.createElement("canvas");
const prepCtx = prep.getContext("2d", { willReadFrequently: true });

/** レターボックス（縦横比を保って縮小し、余白を灰色 114 で埋める）して NCHW の Float32 にする */
export function preprocess(el, w, h, S) {
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

/** 出力 [1, 4+1+K*3, N] から一番自信のある1体を取り出す。戻り値 { conf, box: [x0,y0,x1,y1], xy, kc } か null */
export function postprocess(output, { scale, padX, padY }, confThr) {
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

/** 画像（または video / canvas）を推論する */
export async function detect(session, meta, el, w, h, confThr = 0.25) {
  const pre = preprocess(el, w, h, meta.imgsz);
  const out = await session.run({ [session.inputNames[0]]: pre.tensor });
  return postprocess(out[session.outputNames[0]], pre, confThr);
}
