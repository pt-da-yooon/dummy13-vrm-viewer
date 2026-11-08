"""VRChat への OSC 送信テスト。

VRChat の設定で OSC を有効にしてから実行する
（アクションメニュー → Options → OSC → Enabled）。

使い方:
    python osc/send_test.py --param TestFloat
アバターに同名の float パラメータがあれば、0→1→0 と変化する。
"""

import argparse
import math
import time

from pythonosc.udp_client import SimpleUDPClient


def main() -> None:
    parser = argparse.ArgumentParser(description="VRChat OSC 送信テスト")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000, help="VRChat の受信ポート（既定 9000）")
    parser.add_argument("--param", default="TestFloat", help="送信するアバターパラメータ名")
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()

    client = SimpleUDPClient(args.host, args.port)
    address = f"/avatar/parameters/{args.param}"
    print(f"{address} に送信中（{args.seconds}秒）… Ctrl+C で終了")

    start = time.time()
    try:
        while time.time() - start < args.seconds:
            t = time.time() - start
            value = (math.sin(t * 2.0) + 1.0) / 2.0  # 0.0〜1.0
            client.send_message(address, float(value))
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass

    client.send_message(address, 0.0)
    print("終了")


if __name__ == "__main__":
    main()
