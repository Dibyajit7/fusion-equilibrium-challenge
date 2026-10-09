"""
Fingerprint every locally downloaded DIII-D training shot, so we can guarantee a
held-out evaluation set never overlaps with ANY shot that has ever sat on this
disk (and could therefore have been used for training).

Why this exists: hf_local_data/ was built by scripts/download_shots.py using
`random.sample()` over the full 7041-shot diii_d_train pool -- a random subset,
not a sequential prefix. That means `local_score.py --skip N` (which skips N
positions in the Hub's *canonical streaming order*) has no guaranteed
relationship to which shots are physically present in hf_local_data/. A
"held-out" evaluation picked by stream position can silently re-score shots
the model was actually trained on.

The fingerprint is a sha1 of each shot's `efit_times` array, which is a
shot-specific real-valued timestamp sequence (its length and exact values
differ per shot) -- read identically whether the shot comes from a local
parquet file or a streamed Hugging Face row, since both trace back to the
same underlying column.

Usage:
    uv run python scripts/compute_local_shot_hashes.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
D3D_LOCAL_DIR = REPO_ROOT / "hf_local_data" / "data" / "diii_d_train"
OUT_PATH = REPO_ROOT / "results" / "local_shot_hashes.json"


def fingerprint(efit_times) -> str:
    arr = np.asarray(efit_times, dtype=np.float64).round(6)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def main():
    files = sorted(D3D_LOCAL_DIR.glob("*.parquet"))
    print(f"Fingerprinting {len(files)} local DIII-D shots...")
    hashes = []
    for i, f in enumerate(files):
        row = pd.read_parquet(f).iloc[0]
        hashes.append(fingerprint(row["efit_times"]))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(files)}")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps({"n_local_shots": len(hashes), "hashes": sorted(set(hashes))}))
    print(f"Saved {len(set(hashes))} unique fingerprints -> {OUT_PATH}")


if __name__ == "__main__":
    main()
