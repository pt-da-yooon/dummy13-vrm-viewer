"""synth/ の通常 Python 側スクリプト（composite.py, preview.py）で共有する処理。"""

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
DEFAULT_KEYPOINTS = REPO / "configs" / "keypoints.yaml"


def load_keypoint_config(path: str | Path = DEFAULT_KEYPOINTS) -> dict:
    """keypoints.yaml を読み、名前・骨格（添字）・flip_idx を返す。"""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    names = cfg["keypoints"]
    index = {n: i for i, n in enumerate(names)}
    flip_idx = list(range(len(names)))
    for a, b in cfg.get("flip_pairs", []):
        flip_idx[index[a]], flip_idx[index[b]] = index[b], index[a]
    skeleton = [(index[a], index[b]) for a, b in cfg.get("skeleton", [])]
    return {"names": names, "skeleton": skeleton, "flip_idx": flip_idx}
