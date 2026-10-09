"""
local_score.py, but the held-out shots are guaranteed disjoint from every shot that
has ever been downloaded to hf_local_data/ (and could therefore have been used for
training) -- see scripts/compute_local_shot_hashes.py for why `--skip N` alone does
not guarantee that.

Streams the Hub's diii_d_train split in canonical order, skipping any shot whose
`efit_times` fingerprint is in results/local_shot_hashes.json, until it has collected
--n-shots genuinely unseen ones. Then hands off to local_score.py's own reference
building / prediction / scoring code unchanged.

Usage:
    uv run python scripts/leak_safe_score.py --n-shots 50
    uv run python scripts/leak_safe_score.py --n-shots 50 --start 6000   # scan from further in
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "fusion_scoring"))

from local_score import (  # noqa: E402
    build_reference, finalize_machine, predict, psi_residuals, score_shot, MACHINE, N_CONS,
    N_SCALARS, SCALARS, PSI_SIGNS, AXIS_SIGN,
)
from scripts.compute_local_shot_hashes import fingerprint  # noqa: E402
from common import SCORING_VERSION  # noqa: E402
from metrics import Accum  # noqa: E402

REPO_ID = "Sophelio/fusion-equilibrium-challenge"
HASHES_PATH = REPO_ROOT / "results" / "local_shot_hashes.json"


def load_held_out_shots(n_shots: int, start: int) -> list[dict]:
    from datasets import load_dataset

    local_hashes = set(json.loads(HASHES_PATH.read_text())["hashes"])
    print(f"Excluding {len(local_hashes)} fingerprints seen in hf_local_data/")

    ds = load_dataset(REPO_ID, "diii_d_train", split="train", streaming=True)
    out, n_skipped_pos, n_excluded = [], 0, 0
    for i, row in enumerate(ds):
        if i < start:
            continue
        n_skipped_pos += 1
        fp = fingerprint(row["efit_times"])
        if fp in local_hashes:
            n_excluded += 1
            continue
        psi = np.asarray(row["efit_psirz"], dtype=np.float32)
        if psi.ndim != 3:
            psi = np.stack([np.asarray(f, dtype=np.float32) for f in row["efit_psirz"]])
        out.append({
            "row": row, "psi": psi,
            "q95": np.asarray(row["efit_q95"], dtype=np.float64),
            "betaN": np.asarray(row["efit_beta_n"], dtype=np.float64),
        })
        print(f"  loaded held-out shot {len(out) - 1} (stream pos {i})  T={psi.shape[0]}")
        if len(out) >= n_shots:
            break
    if not out:
        raise SystemExit("No held-out shots found -- increase --start or re-check hashes.")
    print(f"Scanned {n_skipped_pos} stream positions from pos {start}, "
          f"excluded {n_excluded} as locally-seen, kept {len(out)} guaranteed-unseen shots.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-shots", type=int, default=50)
    ap.add_argument("--start", type=int, default=0, help="stream position to start scanning from")
    ap.add_argument("--pred", type=Path, help="score this .npz instead of calling your model")
    args = ap.parse_args()

    mask = np.load(REPO_ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    R, Z = mask["grid_R"], mask["grid_Z"]
    mask_coarse = mask["mask_coarse"].astype(bool)
    mask_f = mask_coarse.astype(np.float64)

    print(f"Fusion Equilibrium Challenge -- leak-safe local scorer "
          f"(metric v{SCORING_VERSION}, {MACHINE})")
    shots = load_held_out_shots(args.n_shots, args.start)

    print("Building reference targets from ground truth...")
    refs, psi_sum, psi_sumsq, psi_n = [], 0.0, 0.0, 0.0
    scal_sum, scal_sumsq, scal_n = np.zeros(N_SCALARS), np.zeros(N_SCALARS), np.zeros(N_SCALARS)
    cons_sum, cons_sumsq, cons_n = np.zeros(N_CONS), np.zeros(N_CONS), np.zeros(N_CONS)
    for si, s in enumerate(shots):
        ref = build_reference(s["psi"], R, Z, mask_coarse, mask_f)
        refs.append(ref)
        g = s["psi"].astype(np.float64)
        fin = np.isfinite(g)
        psi_sum += float(g[fin].sum()); psi_sumsq += float((g[fin] ** 2).sum()); psi_n += int(fin.sum())
        for j, name in enumerate(SCALARS):
            v = np.asarray(s[name], dtype=np.float64)
            v = v[np.isfinite(v)]
            scal_sum[j] += v.sum(); scal_sumsq[j] += (v ** 2).sum(); scal_n[j] += v.size
        cons_gt, cmask = ref[1], ref[2]
        for j in range(N_CONS):
            v = cons_gt[cmask[:, j], j]
            cons_sum[j] += v.sum(); cons_sumsq[j] += (v ** 2).sum(); cons_n[j] += v.size
        print(f"  shot {si}: reference built")

    ref_stats = {
        "psi_sum": psi_sum, "psi_sumsq": psi_sumsq, "psi_n": psi_n,
        "scal_sum": scal_sum, "scal_sumsq": scal_sumsq, "scal_n": scal_n,
        "cons_sum": cons_sum, "cons_sumsq": cons_sumsq, "cons_n": cons_n,
    }
    mean_psi = psi_sum / psi_n if psi_n else 0.0
    means = (mean_psi,
             np.where(scal_n > 0, scal_sum / np.where(scal_n > 0, scal_n, 1), 0.0),
             np.where(cons_n > 0, cons_sum / np.where(cons_n > 0, cons_n, 1), 0.0))

    print("Generating predictions...")
    preds = predict(shots, "model", args.pred)

    totals = {s: 0.0 for s in PSI_SIGNS}
    for s_, p in zip(shots, preds):
        rr, _, _ = psi_residuals(s_["psi"], p["psirz"], mean_psi)
        for sg in PSI_SIGNS:
            totals[sg] += rr[sg]
    psi_sign = min(PSI_SIGNS, key=lambda sg: totals[sg])
    if psi_sign < 0:
        print("  note: flux is sign-inverted vs DIII-D convention -- normalized for you")

    print("Scoring...")
    acc = Accum()
    acc.psi_sign = psi_sign
    for si, (s_, ref, p) in enumerate(zip(shots, refs, preds)):
        acc.add(score_shot(s_, ref, p, R, Z, mask_coarse, mask_f, psi_sign, means))
        print(f"  shot {si} scored")

    res = finalize_machine(acc, ref_stats)

    print("\n" + "=" * 62)
    print(f"  COMPOSITE S = {res['S']:.4f}      ({len(shots)} LEAK-SAFE held-out {MACHINE} shots)")
    print("=" * 62)
    for label, key, w in [("R2_psi", "r2_psi", 0.55), ("R2_{q95,betaN}", "r2_qb", 0.15),
                          ("1 - D_LCFS", "dlcfs", 0.10), ("Consistency", "consistency", 0.20)]:
        v = (1.0 - min(1.0, res["dlcfs"])) if key == "dlcfs" else res[key]
        print(f"  {label:>16s}  {v:8.4f}   x {w:.2f}  =  {w * max(0.0, v):.4f}")
    print(f"\n  per-derived-scalar R2 (the Consistency term):")
    for k, v in res["r2_cons_each"].items():
        print(f"      {k:>8s}  {'  n/a' if v is None else f'{v:7.4f}'}")
    print(f"\n  LCFS extraction failed on {res['lcfs_fail_frac']:.1%} of frames; "
          f"derivations on {res['cons_fail_frac']:.1%}")
    print(f"  psi_sign = {res['psi_sign']:+d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
