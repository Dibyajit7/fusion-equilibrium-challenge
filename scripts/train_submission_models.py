"""
Train and save the models that submission_skeleton.py's your_model_predict()
loads at inference time: one pipeline per machine.

  DIII-D (Challenge 1): raw coil currents + Thomson scattering -> MLP on PCA
  coefficients for psirz, plus separate MLP regressors for q95/betaN. This is
  the already-validated best pipeline (R2=0.995, SSIM=0.997 at 640 shots).

  MAST (Challenge 2): machine-agnostic physics features (Ip, TF-coil proxy,
  q~TF/Ip, Thomson pressure stats) -> MLP on PCA coefficients for
  PER-FRAME-NORMALIZED psirz shape, plus MLP regressors for q95/betaN on the
  same physics features. MAST has zero training targets by design (true
  zero-shot), so this is trained entirely on DIII-D and applied cold.

  Absolute-scale recovery for MAST: the physics pipeline predicts normalized
  SHAPE only. We calibrate back to physical units using the empirical
  (mean, std) of the 3 bundled MAST demo shots (115 frames total) -- the only
  real MAST flux values available anywhere locally -- and flip sign, since
  the dataset card states MAST's flux maximum sits at the magnetic axis where
  DIII-D's sits at the minimum (our shape model is trained purely on DIII-D's
  convention). This is a crude, machine-level constant, not a per-shot
  adaptive calibration -- see reports/team-log.md for the honest limitation.

Usage:
    uv run python scripts/train_submission_models.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from experiments import (  # noqa: E402
    D3D_MAGNETICS_SIGNALS,
    EFIT_SCALAR_LABELS,
    TargetPCA,
    build_feature_matrix,
    load_shot_from_hf_row,
)
from cross_machine_physics_features import (  # noqa: E402
    MAST_DEMO_FILES,
    build_physics_dataset,
    load_rows,
    normalize_flux_per_frame,
)

D3D_LOCAL_DIR = REPO_ROOT / "hf_local_data" / "data" / "diii_d_train"
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

Q95_IDX = EFIT_SCALAR_LABELS.index("efit_q95")
BETAN_IDX = EFIT_SCALAR_LABELS.index("efit_beta_n")

# Lean column set for the D3D stage -- only the columns this pipeline actually
# reads, not every column a full-row load would pull in. Same NEEDED_COLUMNS
# pattern as scripts/train_torch_d3d.py, and for the same reason: reading+
# holding every row's full column set for ALL shots at once (the previous
# approach here) caused a real jetsam OOM kill once shot counts got large.
# Processing one lean shot at a time instead keeps at most one shot's data
# resident, regardless of how many shots are trained on.
D3D_NEEDED_COLUMNS = (
    ["source", "efit_times", "efit_psirz", "magnetics_time", "magnetics_plasma_current_times"]
    + [f"magnetics_{sig}" for sig in D3D_MAGNETICS_SIGNALS]
    + ["thomson_core_times", "thomson_core_Te", "thomson_core_ne",
       "thomson_edge_times", "thomson_edge_Te", "thomson_edge_ne"]
    + list(EFIT_SCALAR_LABELS)
)


def load_shot_lean(path: Path) -> dict:
    row = pd.read_parquet(path, columns=D3D_NEEDED_COLUMNS).iloc[0]
    return load_shot_from_hf_row(row)


def fit_scalar_model(X: np.ndarray, y: np.ndarray) -> MLPRegressor:
    mask = np.isfinite(y)
    model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000,
                          early_stopping=True, random_state=42)
    model.fit(X[mask], y[mask])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-shots", type=int, default=0,
                        help="cap the number of local DIII-D shots used (0 = all available)")
    parser.add_argument("--stage", choices=["d3d", "mast", "both"], default="both",
                        help="run only one pipeline (each saves its own artifacts and can be "
                             "run independently, so a kill mid-run doesn't lose the other stage)")
    args = parser.parse_args()

    d3d_files = sorted(D3D_LOCAL_DIR.glob("*.parquet"))
    if args.max_shots:
        d3d_files = d3d_files[:args.max_shots]
    print(f"Training on {len(d3d_files)} local DIII-D shots")

    if args.stage in ("d3d", "both"):
        train_d3d(d3d_files)
    if args.stage in ("mast", "both"):
        train_mast(d3d_files)

    print("\nDone. All artifacts in models/ -- see submission_skeleton.py's "
          "your_model_predict() for how they're loaded and used.")


def train_d3d(d3d_files):
    # ------------------------------------------------------------------
    # DIII-D pipeline (raw coils + Thomson)
    # ------------------------------------------------------------------
    print("\n=== DIII-D pipeline ===")

    # include_thomson=True: raw per-channel Thomson expansion used to break
    # concatenation into one fixed-width feature matrix, since channel count
    # varies shot-to-shot (44 vs 42 vs 54 core channels -- genuine hardware/
    # config differences, not a bug). experiments.interpolate_thomson_to_efit
    # now reduces each profile to shape-agnostic per-timestep stats (mean/std/
    # max/min + electron pressure) instead of indexing channels by position,
    # which is a fixed width regardless of channel count -- see its docstring.
    # See reports/team-log.md for the original bug.
    #
    # One shot loaded (lean columns) at a time, immediately reduced to its
    # small feature/target slice -- see D3D_NEEDED_COLUMNS above for why.
    t0 = time.time()
    X_parts, Y_parts, S_parts = [], [], []
    for i, f in enumerate(d3d_files):
        shot = load_shot_lean(f)
        shot["shot_index"] = i
        Xi, Yi, Si, _ = build_feature_matrix([shot], include_thomson=True)
        X_parts.append(Xi)
        Y_parts.append(Yi)
        S_parts.append(Si)
        del shot
        if (i + 1) % 200 == 0:
            print(f"  loaded {i + 1}/{len(d3d_files)} shots ({time.time() - t0:.0f}s)")
    X = np.concatenate(X_parts)
    Y = np.concatenate(Y_parts)
    S = np.concatenate(S_parts)
    del X_parts, Y_parts, S_parts
    print(f"  X: {X.shape}, Y: {Y.shape}")

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    pca = TargetPCA(n_components=50).fit(Y)
    print(f"  PCA(50): {np.cumsum(pca.explained_variance_ratio)[-1] * 100:.2f}% variance")

    psi_model = MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=500,
                              early_stopping=True, random_state=42)
    psi_model.fit(X_scaled, pca.transform(Y))
    print("  psi_model trained")

    q95_model = fit_scalar_model(X_scaled, S[:, Q95_IDX])
    betaN_model = fit_scalar_model(X_scaled, S[:, BETAN_IDX])
    print("  q95_model, betaN_model trained")

    joblib.dump(scaler, MODELS_DIR / "d3d_scaler.joblib")
    joblib.dump(pca, MODELS_DIR / "d3d_pca.joblib")
    joblib.dump(psi_model, MODELS_DIR / "d3d_psi_model.joblib")
    joblib.dump(q95_model, MODELS_DIR / "d3d_q95_model.joblib")
    joblib.dump(betaN_model, MODELS_DIR / "d3d_betan_model.joblib")
    print(f"  Saved to {MODELS_DIR}/d3d_*.joblib")


def train_mast(d3d_files):
    # ------------------------------------------------------------------
    # MAST pipeline (machine-agnostic physics features, trained on DIII-D)
    # ------------------------------------------------------------------
    print("\n=== MAST pipeline (physics features, zero-shot) ===")

    # Physics features need Thomson (pressure stats), so this reads full rows
    # -- but still one row at a time, discarded immediately after extracting
    # its small feature/target slice, so peak memory stays at ~1 row.
    t0 = time.time()
    Xp_parts, Yp_parts, id_parts, scalars_list = [], [], [], []
    for i, f in enumerate(d3d_files):
        row = pd.read_parquet(f).iloc[0]
        Xp_i, Yp_i, _ = build_physics_dataset([row])
        Xp_parts.append(Xp_i)
        Yp_parts.append(Yp_i)
        id_parts.append(np.full(len(Xp_i), i, dtype=np.int32))
        shot = load_shot_from_hf_row(row)
        scalars_list.append(shot.get("scalars", {}))
        del row, shot
        if (i + 1) % 200 == 0:
            print(f"  loaded {i + 1}/{len(d3d_files)} shots ({time.time() - t0:.0f}s)")
    Xp = np.concatenate(Xp_parts)
    Yp_raw = np.concatenate(Yp_parts)
    shot_ids_p = np.concatenate(id_parts)
    del Xp_parts, Yp_parts, id_parts
    print(f"  Xp: {Xp.shape}, Yp: {Yp_raw.shape}")
    Yp_norm, _, _ = normalize_flux_per_frame(Yp_raw)

    physics_scaler = StandardScaler().fit(Xp)
    Xp_scaled = physics_scaler.transform(Xp)

    physics_pca = TargetPCA(n_components=30).fit(Yp_norm)
    physics_psi_model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500,
                                      early_stopping=True, random_state=42)
    physics_psi_model.fit(Xp_scaled, physics_pca.transform(Yp_norm))
    print("  physics_psi_model trained (predicts normalized shape)")

    # Scalars via the same physics features, so they're computable on MAST too
    # (the D3D raw+Thomson scalar models above use D3D-only column names).
    S_phys = np.full((len(Xp), 2), np.nan, dtype=np.float32)
    offset = 0
    for shot_id in np.unique(shot_ids_p):
        n = int((shot_ids_p == shot_id).sum())
        shot_scalars = scalars_list[shot_id]
        for j, name in enumerate(["efit_q95", "efit_beta_n"]):
            arr = shot_scalars.get(name)
            if arr is not None:
                m = min(n, len(arr))
                S_phys[offset:offset + m, j] = arr[:m]
        offset += n

    physics_q95_model = fit_scalar_model(Xp_scaled, S_phys[:, 0])
    physics_betaN_model = fit_scalar_model(Xp_scaled, S_phys[:, 1])
    print("  physics_q95_model, physics_betaN_model trained")

    joblib.dump(physics_scaler, MODELS_DIR / "physics_scaler.joblib")
    joblib.dump(physics_pca, MODELS_DIR / "physics_pca.joblib")
    joblib.dump(physics_psi_model, MODELS_DIR / "physics_psi_model.joblib")
    joblib.dump(physics_q95_model, MODELS_DIR / "physics_q95_model.joblib")
    joblib.dump(physics_betaN_model, MODELS_DIR / "physics_betan_model.joblib")
    print(f"  Saved to {MODELS_DIR}/physics_*.joblib")

    # ------------------------------------------------------------------
    # MAST absolute-scale calibration constants (from the 3 demo shots --
    # the only real MAST flux values available locally)
    # ------------------------------------------------------------------
    print("\n=== MAST calibration constants ===")
    mast_rows = load_rows(MAST_DEMO_FILES)
    _, Y_mast_raw, _ = build_physics_dataset(mast_rows)
    mast_mean = float(Y_mast_raw.mean())
    mast_std = float(Y_mast_raw.std())
    print(f"  MAST demo shots (n={len(Y_mast_raw)} frames): mean={mast_mean:.4f}, "
          f"std={mast_std:.4f}")

    calibration = {"mast_mean": mast_mean, "mast_std": mast_std, "mast_sign": -1.0}
    joblib.dump(calibration, MODELS_DIR / "mast_calibration.joblib")
    print(f"  Saved to {MODELS_DIR}/mast_calibration.joblib")


if __name__ == "__main__":
    main()
