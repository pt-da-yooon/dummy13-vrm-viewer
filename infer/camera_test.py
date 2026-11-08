"""Webカメラが映るかだけを確認する（学習済みモデルは不要）。

使い方:
    python infer/camera_test.py            # 使えるカメラ番号を探して、最初の1台を表示
    python infer/camera_test.py --camera 1 # 番号を指定
q で終了。
"""

import argparse
import sys

import cv2


def open_camera(i: int) -> cv2.VideoCapture:
    # Windows では DirectShow の方が早く開けることが多い
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(i)
    return cap


def main() -> None:
    p = argparse.ArgumentParser(description="Webカメラの確認")
    p.add_argument("--camera", type=int, help="カメラ番号（省略時は 0〜4 を探す）")
    args = p.parse_args()

    if args.camera is None:
        found = []
        for i in range(5):
            cap = open_camera(i)
            ok, frame = cap.read() if cap.isOpened() else (False, None)
            if ok:
                found.append(i)
                print(f"カメラ {i}: 使える（{frame.shape[1]}x{frame.shape[0]}）")
            cap.release()
        if not found:
            raise SystemExit("カメラが見つかりません（接続・他のアプリでの使用中・プライバシー設定を確認）")
        cam = found[0]
    else:
        cam = args.camera

    cap = open_camera(cam)
    if not cap.isOpened():
        raise SystemExit(f"カメラ {cam} を開けません")
    print(f"カメラ {cam} を表示中… q で終了")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("映像が取れません")
            break
        cv2.putText(frame, f"camera {cam}  {frame.shape[1]}x{frame.shape[0]}  (q: quit)", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("camera test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
