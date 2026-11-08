"""Webカメラで Dummy13 の姿勢を推定して表示し、必要なら VRChat へ OSC で送る。

使い方:
    python infer/webcam.py                                   # runs/pose/dummy13/weights/best.pt を使う
    python infer/webcam.py --weights path/to/best.pt --camera 1
    python infer/webcam.py --source video.mp4                # 動画ファイルで試す
    python infer/webcam.py --osc                             # 関節角度を VRChat に送る

画面: 緑=信頼度の高いキーポイント, 橙=低い。q で終了。

OSC について（まだ試作）:
    2D 画像上の関節の曲がり具合（0〜180°）を 0〜1 の float にして
    /avatar/parameters/D13_<名前> に送る。アバター側に同名の float パラメータと
    それを使う Animator が必要。3D 化（奥行き方向の曲げ）はまだしていない。
"""

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent

# 2D の関節角度: 名前 → (端点 a, 関節, 端点 b)。a-関節-b のなす角。
ANGLES = {
    "LElbow": ("l_shoulder", "l_elbow", "l_wrist"),
    "RElbow": ("r_shoulder", "r_elbow", "r_wrist"),
    "LKnee": ("l_hip", "l_knee", "l_ankle"),
    "RKnee": ("r_hip", "r_knee", "r_ankle"),
    "LShoulder": ("neck", "l_shoulder", "l_elbow"),
    "RShoulder": ("neck", "r_shoulder", "r_elbow"),
    "LHip": ("hips", "l_hip", "l_knee"),
    "RHip": ("hips", "r_hip", "r_knee"),
}


class OneEuro:
    """ジッター除去用の One Euro フィルタ（動きが遅いときは強く、速いときは弱く平滑化）。"""

    def __init__(self, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        self.dx = self.dx + self._alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        self.x = self.x + self._alpha(cutoff, dt) * (x - self.x)
        self.t = t
        return self.x


def joint_angle(a, b, c) -> float:
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-6:
        return float("nan")
    return math.degrees(math.acos(np.clip(v1 @ v2 / n, -1, 1)))


def main() -> None:
    p = argparse.ArgumentParser(description="Dummy13 ポーズ推定（Webカメラ）")
    p.add_argument("--weights", default=str(REPO / "runs" / "pose" / "dummy13" / "weights" / "best.pt"))
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--source", help="カメラの代わりに動画・画像ファイル")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.4, help="フィギュア検出の信頼度のしきい値")
    p.add_argument("--kpt-conf", type=float, default=0.5, help="キーポイントを使う信頼度のしきい値")
    p.add_argument("--osc", action="store_true", help="VRChat に OSC で送る")
    p.add_argument("--osc-host", default="127.0.0.1")
    p.add_argument("--osc-port", type=int, default=9000)
    p.add_argument("--no-window", action="store_true")
    args = p.parse_args()

    kcfg = yaml.safe_load((REPO / "configs" / "keypoints.yaml").read_text(encoding="utf-8"))
    names = kcfg["keypoints"]
    idx = {n: i for i, n in enumerate(names)}
    skeleton = [(idx[a], idx[b]) for a, b in kcfg["skeleton"]]

    model = YOLO(args.weights)
    if args.source:
        cap = cv2.VideoCapture(args.source)
    else:  # Windows では DirectShow の方が早く開けることが多い
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("カメラ（または動画）を開けません")

    client = None
    if args.osc:
        from pythonosc.udp_client import SimpleUDPClient
        client = SimpleUDPClient(args.osc_host, args.osc_port)

    filt = OneEuro()
    fps_t, fps = time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        now = time.time()
        if res.keypoints is not None and len(res.keypoints) > 0:
            best = int(res.boxes.conf.argmax())  # 一番自信のある1体だけ使う
            xy = res.keypoints.xy[best].cpu().numpy()
            kc = res.keypoints.conf[best].cpu().numpy() if res.keypoints.conf is not None else np.ones(len(xy))
            xy = filt(xy, now)
            good = kc >= args.kpt_conf
            for a, b in skeleton:
                if good[a] and good[b]:
                    cv2.line(frame, tuple(map(int, xy[a])), tuple(map(int, xy[b])), (230, 230, 230), 2, cv2.LINE_AA)
            for i, (x, y) in enumerate(xy):
                cv2.circle(frame, (int(x), int(y)), 4, (0, 220, 0) if good[i] else (0, 160, 255), -1, cv2.LINE_AA)
            angles = {}
            for name, (a, b, c) in ANGLES.items():
                if good[idx[a]] and good[idx[b]] and good[idx[c]]:
                    angles[name] = joint_angle(xy[idx[a]], xy[idx[b]], xy[idx[c]])
            y0 = 24
            for name, deg in angles.items():
                cv2.putText(frame, f"{name}: {deg:5.1f}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y0 += 18
                if client and not math.isnan(deg):
                    client.send_message(f"/avatar/parameters/D13_{name}", float(deg / 180.0))
        fps = 0.9 * fps + 0.1 / max(now - fps_t, 1e-3)
        fps_t = now
        cv2.putText(frame, f"{fps:4.1f} fps", (frame.shape[1] - 90, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if not args.no_window:
            cv2.imshow("Dummy13 pose", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
