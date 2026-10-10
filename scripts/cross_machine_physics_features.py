"""
Cross-machine (DIII-D -> MAST) transfer experiment using machine-agnostic
physics features, per MODELING_GUIDE.md's "Synthetic Diagnostics:
Machine-Agnostic Inputs" section.

The official mast_public_test config withholds all targets by design (this is
a zero-shot challenge -- there is no mast_train config at all). The only MAST
shots with local ground truth are the 3 demo shots bundled in parquet_data/
for the dFL visualizer. That is the full extent of what can be checked
locally before spending a Codabench submission slot -- treat the MAST numbers
below as a directional sanity check, not a statistically meaningful score.

Compares two approaches, trained only on DIII-D, evaluated zero-shot on MAST:

  A) RAW      -- DIII-D-named coil columns (experiments.py's existing
                 build_feature_matrix). Zero-filled for MAST, since MAST
                 doesn't have those columns under those names. This is the
                 "naive" approach the challenge paper reports collapsing
                 from SSIM 0.83 (DIII-D) to SSIM 0.10 (MAST).

  B) PHYSICS  -- machine-agnostic features (Ip, TF-coil-current proxy,
                 q ~ TF/Ip, Thomson electron-pressure profile stats) +
                 per-frame normalized flux targets (zero-mean/unit-std per
                 65x65 frame), so the model predicts *shape* instead of
                 absolute scale. MAST's flux has its maximum at the magnetic
                 axis where DIII-D has its minimum (dataset card, "Sign
                 convention difference") -- the official scorer "normalizes
                 global flux sign" before scoring, so this script does the
                 same: SSIM is taken as the best of the direct and
                 sign-flipped comparison.

Usage:
    uv run python scripts/cross_machine_physics_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data_fixes import fix_d3d_ip_times  # noqa: E402
from experiments import (  # noqa: E402
    D3D_MAGNETICS_SIGNALS,
    TargetPCA,
    _as_psirz_stack,
    build_feature_matrix,
    compute_metrics,
    load_shot_from_hf_row,
)

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False

D3D_LOCAL_DIR = REPO_ROOT / "hf_local_data" / "data" / "diii_d_train"
MAST_DEMO_FILES = sorted((REPO_ROOT / "parquet_data").glob("mast_shot_*.parquet"))
D3D_DEMO_FILES = sorted((REPO_ROOT / "parquet_data").glob("d3d_shot_*.parquet"))

EPS = 1e-8


# ---------------------------------------------------------------------------
# Machine-agnostic physics feature extraction
# ---------------------------------------------------------------------------

def _interp(times, values, target_times, fill=0.0):
    mask = np.isfinite(values) & np.isfinite(times)
    if mask.sum() < 2:
        return np.full(len(target_times), fill, dtype=np.float32)
    f = interp1d(times[mask], values[mask], kind="linear",
                 fill_value=fill, bounds_error=False)
    return f(target_times).astype(np.float32)


def _thomson_pressure_stats(row: dict, target_times: np.ndarray) -> np.ndarray:
    """pe = ne * Te at each Thomson-core timestep, reduced to 4 shape-agnostic
    scalars per timestep, then interpolated onto target_times. Units (eV, m^-3)
    are the same physical quantities on both machines -- genuinely machine-agnostic,
    unlike raw per-coil currents."""
    try:
        times = np.asarray(row["thomson_core_times"], dtype=np.float64)
        # float64: ne ~1e19-1e20 m^-3 times Te ~1e2-1e4 eV gives pe ~1e21-1e24,
        # and nanstd squares internally -- pe**2 overflows float32 (max ~3.4e38)
        # but not float64 (max ~1.8e308).
        Te = np.asarray(row["thomson_core_Te"], dtype=np.float64)
        ne = np.asarray(row["thomson_core_ne"], dtype=np.float64)
    except Exception:
        return np.zeros((len(target_times), 4), dtype=np.float32)

    pe = Te * ne  # (n_thomson_times, n_channels)
    with np.errstate(invalid="ignore"):
        peak = np.nanmax(pe, axis=1)
        integrated = np.nansum(np.where(np.isfinite(pe), pe, 0.0), axis=1)
        mean = np.nanmean(pe, axis=1)
        std = np.nanstd(pe, axis=1)

    out = np.column_stack([
        _interp(times, peak, target_times),
        _interp(times, integrated, target_times),
        _interp(times, mean, target_times),
        _interp(times, std, target_times),
    ])
    return out


def build_physics_features(row: dict, source: str, efit_times: np.ndarray) -> np.ndarray:
    """7 machine-agnostic features per EFIT timestep: Ip, TF-coil current proxy,
    q ~ TF/Ip, and 4 Thomson electron-pressure profile stats."""
    mag_time = np.asarray(row["magnetics_time"], dtype=np.float64)

    if source == "DIII-D":
        ip_times = fix_d3d_ip_times(row)
        tf_col = "magnetics_bcoil"
    else:
        ip_times = mag_time
        tf_col = "magnetics_tf_current"

    ip = _interp(ip_times, np.asarray(row["magnetics_plasma_current"], dtype=np.float32),
                 efit_times)
    tf = _interp(mag_time, np.asarray(row[tf_col], dtype=np.float32), efit_times)
    q_proxy = tf / (np.abs(ip) + EPS)

    pe_stats = _thomson_pressure_stats(row, efit_times)

    return np.column_stack([ip, tf, q_proxy, pe_stats])  # (T, 7)


FEATURE_NAMES = ["Ip", "TF_coil_proxy", "q_proxy_TF_over_Ip",
                 "pe_peak", "pe_integrated", "pe_mean", "pe_std"]


def normalize_flux_per_frame(psirz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame z-score: removes absolute scale AND the DIII-D/MAST amplitude
    difference (dataset card: DIII-D median ~0.61 Wb/rad vs MAST ~0.21), so the
    model learns shape, not magnitude.

    Mutates psirz in place and returns it, rather than allocating a separate
    full-size output array -- at full dataset scale (65x65 x ~1M frames) that
    doubling was large enough on its own to cause real memory exhaustion
    (confirmed via a memory-sampling sidecar during training). No current
    caller needs the pre-normalization values afterward."""
    flat = psirz.reshape(len(psirz), -1)
    mean = flat.mean(axis=1).astype(np.float32)
    std = (flat.std(axis=1) + EPS).astype(np.float32)
    psirz -= mean[:, None, None]
    psirz /= std[:, None, None]
    return psirz.astype(np.float32, copy=False), mean, std


def best_sign_ssim(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """SSIM against both signs of y_pred, taking the better match -- mirrors the
    official scorer's documented global flux-sign normalization."""
    if not HAS_SSIM:
        return float("nan")
    vals = []
    for i in range(len(y_true)):
        dr = y_true[i].max() - y_true[i].min()
        s_direct = ssim(y_true[i], y_pred[i], data_range=dr)
        s_flipped = ssim(y_true[i], -y_pred[i], data_range=dr)
        vals.append(max(s_direct, s_flipped))
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_rows(files: list[Path]) -> list[dict]:
    return [pd.read_parquet(f).iloc[0] for f in files]


def load_shot_safe(row) -> dict:
    """load_shot_from_hf_row calls fix_d3d_ip_times() unconditionally, which
    looks up a magnetics_plasma_current_times column that only exists on
    DIII-D rows (MAST has one shared magnetics_time base, no erratum to fix).
    Patch the row with that column before delegating, for MAST only."""
    if row.get("source", "DIII-D") != "DIII-D" and "magnetics_plasma_current_times" not in row:
        row = row.copy()
        row["magnetics_plasma_current_times"] = row["magnetics_time"]
    return load_shot_from_hf_row(row)


def build_physics_dataset(rows: list, max_per_shot: Optional[int] = None):
    X_parts, Y_parts, ids = [], [], []
    for i, row in enumerate(rows):
        source = row.get("source", "DIII-D")
        efit_times = np.asarray(row["efit_times"], dtype=np.float64)
        psirz = _as_psirz_stack(row["efit_psirz"])

        feats = build_physics_features(row, source, efit_times)
        n = min(len(feats), len(psirz))
        if max_per_shot:
            n = min(n, max_per_shot)

        valid = np.isfinite(feats[:n]).all(axis=1)
        X_parts.append(feats[:n][valid])
        Y_parts.append(psirz[:n][valid])
        ids.append(np.full(valid.sum(), i, dtype=np.int32))

    X = np.concatenate(X_parts)
    Y = np.concatenate(Y_parts)
    shot_ids = np.concatenate(ids)
    return X, Y, shot_ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    d3d_files = sorted(D3D_LOCAL_DIR.glob("*.parquet"))
    if not d3d_files:
        raise SystemExit(f"No DIII-D shots found in {D3D_LOCAL_DIR}. Run "
                          f"scripts/download_shots.py first.")
    if not MAST_DEMO_FILES:
        raise SystemExit("No MAST demo shots found in parquet_data/. Run "
                          "`git lfs pull` first.")

    print(f"DIII-D shots: {len(d3d_files)}  |  MAST demo shots (local ground truth): "
          f"{len(MAST_DEMO_FILES)}")

    print("\n=== Loading rows ===")
    d3d_rows = load_rows(d3d_files)
    mast_rows = load_rows(MAST_DEMO_FILES)

    # ----------------------------------------------------------------
    # Approach B: machine-agnostic physics features + normalized targets
    # ----------------------------------------------------------------
    print("\n=== [PHYSICS] Building machine-agnostic feature matrix (DIII-D) ===")
    X, Y_raw, shot_ids = build_physics_dataset(d3d_rows)
    Y_norm, _, _ = normalize_flux_per_frame(Y_raw)
    print(f"  X: {X.shape}, Y_norm: {Y_norm.shape}, unique shots: {len(np.unique(shot_ids))}")

    rng = np.random.RandomState(42)
    unique_shots = np.unique(shot_ids)
    perm = rng.permutation(unique_shots)
    n_test = max(1, int(len(unique_shots) * 0.15))
    test_shots = set(perm[:n_test])
    train_shots = set(perm[n_test:])
    train_mask = np.isin(shot_ids, list(train_shots))
    test_mask = np.isin(shot_ids, list(test_shots))

    scaler = StandardScaler().fit(X[train_mask])
    X_train, X_test = scaler.transform(X[train_mask]), scaler.transform(X[test_mask])
    Y_train_norm, Y_test_norm = Y_norm[train_mask], Y_norm[test_mask]

    n_pca = min(30, len(X_train))
    pca = TargetPCA(n_components=n_pca).fit(Y_train_norm)
    print(f"  PCA({n_pca}) captures {np.cumsum(pca.explained_variance_ratio)[-1] * 100:.1f}% "
          f"of normalized-flux variance")

    model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500,
                          early_stopping=True, random_state=42)
    model.fit(X_train, pca.transform(Y_train_norm))

    # In-domain DIII-D held-out test
    Y_pred_norm = pca.inverse_transform(model.predict(X_test))
    metrics_d3d = compute_metrics(Y_test_norm, Y_pred_norm)
    print(f"\n  [PHYSICS] DIII-D held-out: R2={metrics_d3d['R2']:.4f} "
          f"SSIM={metrics_d3d['SSIM']:.4f}  (normalized-flux space)")

    # Zero-shot MAST (the 3 shots with local ground truth)
    X_mast, Y_mast_raw, _ = build_physics_dataset(mast_rows)
    Y_mast_norm, _, _ = normalize_flux_per_frame(Y_mast_raw)
    X_mast_scaled = scaler.transform(X_mast)  # extrapolating DIII-D's scaler onto MAST
    Y_mast_pred_norm = pca.inverse_transform(model.predict(X_mast_scaled))

    mast_ssim_physics = best_sign_ssim(Y_mast_norm, Y_mast_pred_norm)
    print(f"  [PHYSICS] MAST zero-shot (n={len(Y_mast_norm)} frames, 3 shots): "
          f"SSIM={mast_ssim_physics:.4f}  (sign-canonicalized, normalized-flux space)")

    # ----------------------------------------------------------------
    # Approach A: naive raw DIII-D-named coils (reusing experiments.py as-is)
    # ----------------------------------------------------------------
    print("\n=== [RAW/naive] Building DIII-D-named coil feature matrix ===")
    d3d_shots = [load_shot_safe(r) for r in d3d_rows]
    mast_shots = [load_shot_safe(r) for r in mast_rows]
    for i, s in enumerate(d3d_shots + mast_shots):
        s["shot_index"] = i

    X_raw, Y_raw_flux, _, raw_ids = build_feature_matrix(d3d_shots, include_thomson=False)
    raw_unique = np.unique(raw_ids)
    raw_perm = rng.permutation(raw_unique)
    raw_test = set(raw_perm[:max(1, int(len(raw_unique) * 0.15))])
    raw_train = set(raw_perm) - raw_test
    raw_train_mask = np.isin(raw_ids, list(raw_train))
    raw_test_mask = np.isin(raw_ids, list(raw_test))

    raw_scaler = StandardScaler().fit(X_raw[raw_train_mask])
    Xr_train = raw_scaler.transform(X_raw[raw_train_mask])
    Xr_test = raw_scaler.transform(X_raw[raw_test_mask])
    Yr_train, Yr_test = Y_raw_flux[raw_train_mask], Y_raw_flux[raw_test_mask]

    raw_pca = TargetPCA(n_components=min(30, len(Xr_train))).fit(Yr_train)
    raw_model = RidgeCV(alphas=np.logspace(-3, 3, 10))
    raw_model.fit(Xr_train, raw_pca.transform(Yr_train))

    Yr_pred = raw_pca.inverse_transform(raw_model.predict(Xr_test))
    raw_metrics_d3d = compute_metrics(Yr_test, Yr_pred)
    print(f"\n  [RAW] DIII-D held-out: SSIM={raw_metrics_d3d['SSIM']:.4f}  (raw flux units)")

    X_mast_raw, Y_mast_flux_raw, _, _ = build_feature_matrix(mast_shots, include_thomson=False)
    Xr_mast = raw_scaler.transform(X_mast_raw)
    Yr_mast_pred = raw_pca.inverse_transform(raw_model.predict(Xr_mast))
    raw_mast_ssim = best_sign_ssim(Y_mast_flux_raw, Yr_mast_pred)
    print(f"  [RAW] MAST zero-shot (n={len(Y_mast_flux_raw)} frames): "
          f"SSIM={raw_mast_ssim:.4f}  (raw flux units, sign-canonicalized)")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CROSS-MACHINE TRANSFER SUMMARY (indicative -- only 3 MAST shots have")
    print("local ground truth; the real mast_public_test targets are withheld)")
    print("=" * 70)
    print(f"{'Approach':<12} {'DIII-D SSIM':>14} {'MAST SSIM':>12} {'Transfer ratio':>16}")
    print(f"{'RAW coils':<12} {raw_metrics_d3d['SSIM']:>14.4f} {raw_mast_ssim:>12.4f} "
          f"{raw_mast_ssim / (raw_metrics_d3d['SSIM'] + EPS):>16.4f}")
    print(f"{'PHYSICS':<12} {metrics_d3d['SSIM']:>14.4f} {mast_ssim_physics:>12.4f} "
          f"{mast_ssim_physics / (metrics_d3d['SSIM'] + EPS):>16.4f}")
    print("\nNote: PHYSICS numbers are in per-frame-normalized flux space (shape only);")
    print("RAW numbers are in raw flux units -- the two SSIM columns are not directly")
    print("comparable to each other, only each column's own DIII-D-vs-MAST ratio is.")


if __name__ == "__main__":
    main()
