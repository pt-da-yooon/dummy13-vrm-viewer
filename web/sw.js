// オフラインでも動かすための Service Worker。
// 一度開いたときに読み込んだもの（このサイトのファイル・モデル、CDN のライブラリ）を保存しておき、
// 通信できないときは保存したものを使う。
// - このサイトのファイル：通信できればいつも新しいものを取りに行き（更新がすぐ反映される）、保存も更新する
// - CDN のライブラリ：URL に版が入っていて中身が変わらないので、保存したものを先に使う
const CACHE = "dummy13-viewer-v1";
const CDN = ["https://cdn.jsdelivr.net/"];

// 最初に開いたときに、まとめて保存しておくもの（無いファイルは飛ばす）。ONNX Runtime の wasm などは使ったときに保存される
const PRECACHE = [
  "./", "index.html", "capture.html", "style.css", "app.js", "avatar.js", "i18n.js", "favicon.svg",
  "fonts/NotoSansJP-Black-latin.woff2",
  "models/model.json", "models/depth.json", "models/lifter.json",
  // 大きいモデル（.onnx）はここでは取らない。app.js が読み込むときに同じ置き場に保存する（最初の訪問で二重に取らないため）
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.all.min.js",
  "https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js",
  "https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.core.js",
  "https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/controls/OrbitControls.js",
  "https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/loaders/GLTFLoader.js",
  "https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/utils/BufferGeometryUtils.js",
  "https://cdn.jsdelivr.net/npm/@pixiv/three-vrm@3.5.5/lib/three-vrm.module.min.js",
];
self.addEventListener("install", (e) => {
  self.skipWaiting();
  e.waitUntil(caches.open(CACHE).then((c) => Promise.all(PRECACHE.map((u) => c.add(u).catch(() => {})))));
});
self.addEventListener("activate", (e) => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) if (k !== CACHE) await caches.delete(k);
    await self.clients.claim();
  })());
});

async function networkFirst(req) {
  const cache = await caches.open(CACHE);
  try {
    const res = await fetch(req);
    // 304（変わっていない）は app.js が自分で保存したものを使うので、ここでは保存しない
    if (res.ok) cache.put(req, res.clone());
    return res;
  } catch (e) {
    const hit = await cache.match(req, { ignoreSearch: true });
    if (hit) return hit;
    throw e;
  }
}

async function cacheFirst(req) {
  const cache = await caches.open(CACHE);
  const hit = await cache.match(req);
  if (hit) return hit;
  const res = await fetch(req);
  if (res.ok) cache.put(req, res.clone());
  return res;
}

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = req.url;
  if (url.startsWith(self.location.origin)) e.respondWith(networkFirst(req));
  else if (CDN.some((p) => url.startsWith(p))) e.respondWith(cacheFirst(req));
});
