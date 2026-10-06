#!/usr/bin/env python3
"""
Submission skeleton — a runnable, format-correct example of a challenge submission.

This is the executable version of README → "3. What you predict". Run it and read the
printed shapes to see exactly what a submission looks like — no Git LFS / sample data needed
(it streams the public test inputs from Hugging Face).

What you submit (per shot, at each `efit_times` timestamp), grouped per shot in one .npz per config:
    shot_0000_psirz    (T, H, W)   flux map   DIII-D 65x65 / MAST 65x65 (both dense, no NaN region)
    shot_0000_q95      (T,)        edge safety factor
    shot_0000_betaN    (T,)        normalized beta

That is the WHOLE contract (metric v2): the flux map plus the only two scalars a flux map cannot
contain (q95 needs the toroidal field function F(psi), betaN needs the pressure profile p(psi)).
Everything else — the LCFS boundary, magnetic axis (R_axis/Z_axis), elongation, triangularity,
volume and internal inductance (li) — is DERIVED from your submitted
flux map by the scorer, with the same published functionals it applies to the ground-truth flux.
A scalar can only be earned by a psi that implies it: there is no separate scalar head to tune.

The leaderboard score is the composite
    S = 0.55*R2_psi + 0.15*R2_{q95,betaN} + 0.10*(1 - D_LCFS) + 0.20*Consistency
where Consistency is the mean agreement of the seven psi-derived scalars, f(psi_pred) vs
f(psi_gt). A perfect flux map scores D_LCFS = 0 and Consistency = 1 by construction.

To make a real submission, replace `your_model_predict()` with your trained model. The placeholder
here emits zeros of the correct shape (a valid-but-useless submission) so you can confirm the
plumbing before plugging in a model.

This one script does the whole submission: build -> validate -> push to Hugging Face -> write the
zip you upload to Codabench.

    # quick format check, 5 shots, build only
    uv run python submission_skeleton.py --max-shots 5

    # the real thing: every shot, pushed, with submission_pointer.zip ready to upload
    uv run python submission_skeleton.py --max-shots 0 \
        --repo your-username/fusion-eq-predictions --read-token hf_...

Run it once without --read-token to create the repo; Hugging Face cannot scope a token to a repo
that does not exist yet. See README -> "5. Build and submit".
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import numpy as np
from datasets import load_dataset

REPO_ID = "Sophelio/fusion-equilibrium-challenge"
TEST_CONFIGS = [("diii_d_public_test", "public_test"), ("mast_public_test", "public_test")]
# Native flux grid per machine (rows=Z, cols=R). Both machines are a dense, fully finite 65x65
# grid — MAST's upstream EFIT 65x129 grid (65 real R columns interleaved with 64 empty ones) is
# collapsed to a dense 65x65 in the corrected dataset, so there is no central NaN region.
GRID = {"DIII-D": (65, 65), "MAST": (65, 65)}
# The only two submitted scalars (metric v2), each under its own per-shot key (named, not
# positional, to make column mix-ups impossible). Everything else is derived from your flux map.
SCALARS = ["q95", "betaN"]

# ---------------------------------------------------------------------------
# Two pipelines, trained by scripts/train_submission_models.py:
#   DIII-D -- raw coils + Thomson -> MLP on PCA(psirz), direct scalar MLPs.
#   MAST   -- machine-agnostic physics features (zero MAST training data, by
#             design) -> MLP on PCA(normalized-shape psirz), rescaled to
#             absolute units via a constant calibrated from the 3 bundled
#             MAST demo shots (the only real MAST flux values available
#             locally). See reports/team-log.md for why this is a crude,
#             machine-level constant and not a per-shot adaptive fit.
# ---------------------------------------------------------------------------
import joblib  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from experiments import (  # noqa: E402
    D3D_MAGNETICS_SIGNALS,
    _as_profile_stack,
    fix_d3d_ip_times,
    interpolate_magnetics_to_efit,
)
from cross_machine_physics_features import build_physics_features  # noqa: E402


def _build_input_shot(row: dict) -> dict:
    """Like experiments.load_shot_from_hf_row, but for inputs only -- test-split
    rows withhold efit_psirz/scalars entirely, so building those would KeyError
    on a column that's deliberately not there."""
    shot: dict = {"efit_times": np.asarray(row["efit_times"], dtype=np.float64)}
    shared_mag_time = np.asarray(row["magnetics_time"], dtype=np.float64)
    ip_times = fix_d3d_ip_times(row)

    magnetics = {}
    for sig in D3D_MAGNETICS_SIGNALS:
        data_col = f"magnetics_{sig}"
        if data_col not in row:
            continue
        times = ip_times if sig == "plasma_current" else shared_mag_time
        magnetics[sig] = {"values": np.asarray(row[data_col], dtype=np.float32), "times": times}
    shot["magnetics"] = magnetics

    if "thomson_core_times" in row:
        shot["thomson_core"] = {
            "times": np.asarray(row["thomson_core_times"], dtype=np.float64),
            "Te": _as_profile_stack(row["thomson_core_Te"]),
            "ne": _as_profile_stack(row["thomson_core_ne"]),
        }
    if "thomson_edge_times" in row:
        shot["thomson_edge"] = {
            "times": np.asarray(row["thomson_edge_times"], dtype=np.float64),
            "Te": _as_profile_stack(row["thomson_edge_Te"]),
            "ne": _as_profile_stack(row["thomson_edge_ne"]),
        }
    return shot

MODELS_DIR = Path(__file__).resolve().parent / "models"
_models: dict = {}


def _load_models() -> dict:
    if not _models:
        for name in ["d3d_scaler", "d3d_pca", "d3d_psi_model", "d3d_q95_model", "d3d_betan_model",
                      "physics_scaler", "physics_pca", "physics_psi_model",
                      "physics_q95_model", "physics_betan_model", "mast_calibration"]:
            _models[name] = joblib.load(MODELS_DIR / f"{name}.joblib")
    return _models


def _predict_d3d(row: dict) -> dict:
    # Raw coils only, matching scripts/train_submission_models.py -- raw
    # per-channel Thomson expansion hits a real variable-channel-count issue
    # across shots; see reports/team-log.md.
    m = _load_models()
    shot = _build_input_shot(row)
    X = interpolate_magnetics_to_efit(shot)
    X_scaled = m["d3d_scaler"].transform(X)

    psirz = m["d3d_pca"].inverse_transform(m["d3d_psi_model"].predict(X_scaled))
    q95 = m["d3d_q95_model"].predict(X_scaled).astype(np.float32)
    betaN = m["d3d_betan_model"].predict(X_scaled).astype(np.float32)
    return {"psirz": psirz, "q95": q95, "betaN": betaN}


def _predict_mast(row: dict) -> dict:
    m = _load_models()
    efit_times = np.asarray(row["efit_times"], dtype=np.float64)
    X = build_physics_features(row, "MAST", efit_times)
    X_scaled = m["physics_scaler"].transform(X)

    normalized_shape = m["physics_pca"].inverse_transform(m["physics_psi_model"].predict(X_scaled))
    cal = m["mast_calibration"]
    psirz = cal["mast_sign"] * normalized_shape * cal["mast_std"] + cal["mast_mean"]

    q95 = m["physics_q95_model"].predict(X_scaled).astype(np.float32)
    betaN = m["physics_betan_model"].predict(X_scaled).astype(np.float32)
    return {"psirz": psirz.astype(np.float32), "q95": q95, "betaN": betaN}


def your_model_predict(row: dict, source: str) -> dict:
    """Return predictions for this shot, aligned to row['efit_times'], as a dict:
        {"psirz":  (T, H, W) flux map,
         "q95":    (T,),
         "betaN":  (T,)}

    DIII-D uses the raw-coil+Thomson pipeline (R2=0.995, SSIM=0.997 held-out).
    MAST uses the machine-agnostic physics-feature pipeline, since it has zero
    training targets by design (true zero-shot) -- see reports/team-log.md for
    the cross-machine transfer experiment this is built from, and its honest
    limitations (n=3 MAST ground-truth shots for calibration, not statistically
    robust; a constant rather than per-shot scale/sign recovery)."""
    if source == "DIII-D":
        return _predict_d3d(row)
    return _predict_mast(row)


def build_submission(config: str, split: str, out_dir: Path, max_shots: int) -> Path:
    ds = load_dataset(REPO_ID, config, split=split, streaming=True)
    preds: dict[str, np.ndarray] = {}
    n = 0
    for i, row in enumerate(ds):
        if max_shots and i >= max_shots:
            break
        source = row.get("source", "DIII-D")
        T = len(np.asarray(row["efit_times"]))
        H, W = GRID[source]
        out = your_model_predict(row, source)

        assert out["psirz"].shape == (T, H, W), f"{config} shot {i}: psirz {out['psirz'].shape} != {(T, H, W)}"
        # float16 keeps relative precision everywhere and costs ~0.1% of score; the scorer
        # upcasts on read. Do NOT instead round to a fixed number of decimals -- np.round(psi, 3)
        # leaves R2_psi at 0.99997 while destroying ~35% of the MAST Consistency term.
        preds[f"shot_{i:04d}_psirz"] = out["psirz"].astype(np.float16)
        for name in SCALARS:
            arr = np.asarray(out[name])
            assert arr.shape == (T,), f"{config} shot {i}: {name} {arr.shape} != ({T},)"
            preds[f"shot_{i:04d}_{name}"] = arr.astype(np.float32)

        n = i + 1
        if n % 25 == 0:
            print(f"  {config}: {n} shots")

    out_path = out_dir / f"{config}.npz"
    np.savez_compressed(out_path, **preds)
    psh = preds.get("shot_0000_psirz", np.empty(0)).shape
    print(f"  {config}: {n} shots -> {out_path.name}  (e.g. shot_0000_psirz {psh}, "
          f"shot_0000_q95 {preds.get('shot_0000_q95', np.empty(0)).shape})")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build a submission, validate it, push it to Hugging Face, and write the "
                    "pointer zip you upload to Codabench.")
    ap.add_argument("--max-shots", type=int, default=5, help="cap shots per config (0 = all)")
    ap.add_argument("--out", type=Path, default=Path("submission"))
    ap.add_argument("--repo", help="Hugging Face dataset repo to push to, e.g. you/fusion-eq-preds. "
                                   "Given this, the script also pushes and writes the pointer zip.")
    ap.add_argument("--read-token", default=os.environ.get("HF_READ_TOKEN"),
                    help="fine-grained READ token scoped to --repo (or set HF_READ_TOKEN)")
    ap.add_argument("--zip", dest="zip_out", type=Path, default=Path("submission_pointer.zip"),
                    help="pointer zip to upload to Codabench (default: submission_pointer.zip)")
    ap.add_argument("--skip-validate", action="store_true",
                    help="skip the structure check (not recommended -- it is what catches a "
                         "malformed .npz before you spend a submission slot)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("Building submission (placeholder zeros — swap in your_model_predict):")
    written = []
    for config, split in TEST_CONFIGS:
        written.append(build_submission(config, split, args.out, args.max_shots).name)

    # No manifest is written here: the scorer locates predictions by FILENAME, and on the
    # pointer route push_and_write_pointer() writes the real {repo_id, revision, token} one.
    total_mb = sum((args.out / w).stat().st_size for w in written) / 1e6
    print(f"\nWrote {', '.join(written)} to {args.out.resolve()}  ({total_mb:.0f} MB)")

    if not args.skip_validate:
        print("\nValidating structure...")
        from validate_submission import validate
        for config, _ in TEST_CONFIGS:
            if validate(args.out / f"{config}.npz", config, args.max_shots):
                print(f"\nValidation FAILED for {config}. Fix your_model_predict and rebuild; "
                      "nothing was pushed.", file=sys.stderr)
                return 1

    if args.repo:
        print()
        from push_predictions import push_and_write_pointer
        return push_and_write_pointer(args.repo, args.read_token, args.out, args.zip_out)

    print("\nNext: re-run with --repo <you>/fusion-eq-predictions to push and build the zip you\n"
          "      upload to Codabench. See README -> '5. Build and submit'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
