"""
Download a random sample of shot parquet files as whole files (robust against
streaming-mode range-request flakiness on unreliable connections), laid out in
the Hub's local-mode directory structure so `experiments.py --source local`
can read them directly.

Usage:
    python scripts/download_shots.py --config diii_d_train --n 150
    python scripts/download_shots.py --config mast_public_test --n 100 --seed 7
"""

import argparse
import random

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "Sophelio/fusion-equilibrium-challenge"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="diii_d_train",
                         choices=["diii_d_train", "diii_d_public_test", "mast_public_test"])
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-dir", default="hf_local_data")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    api = HfApi()
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    candidates = sorted(f for f in files if args.config in f)

    random.seed(args.seed)
    sample = random.sample(candidates, min(args.n, len(candidates)))

    ok, failed = 0, []
    for i, f in enumerate(sample, 1):
        for attempt in range(args.retries):
            try:
                hf_hub_download(repo_id=REPO_ID, repo_type="dataset",
                                 filename=f, local_dir=args.local_dir)
                ok += 1
                break
            except Exception as e:
                if attempt == args.retries - 1:
                    failed.append((f, str(e)))
        if i % 25 == 0:
            print(f"{i}/{len(sample)} done, {len(failed)} failed so far")

    print(f"DONE: {ok} ok, {len(failed)} failed")
    if failed:
        print("Failures:", failed[:5])


if __name__ == "__main__":
    main()
