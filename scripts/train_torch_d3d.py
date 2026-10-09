"""
Train the PyTorch conv-decoder DIII-D flux-map model (scripts/torch_models.py) on
raw coil + plasma-current features, direct-regressing the 65x65 psi grid (no PCA
step -- see torch_models.py's docstring for why).

Loads local parquet shots the same way scripts/train_submission_models.py does
(experiments.load_shot_from_hf_row + build_feature_matrix), splits by shot into
train/val, trains on GPU (MPS on Apple Silicon / CUDA / CPU fallback), and saves:
    models/d3d_torch_scaler.joblib
    models/d3d_torch_model.pt

Usage:
    uv run python scripts/train_torch_d3d.py --max-shots 1500 --epochs 60
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from experiments import (  # noqa: E402
    D3D_MAGNETICS_SIGNALS, EFIT_SCALAR_LABELS, build_feature_matrix, load_shot_from_hf_row,
)
from torch_models import FluxDecoderNet, resolve_device  # noqa: E402

D3D_LOCAL_DIR = REPO_ROOT / "hf_local_data" / "data" / "diii_d_train"
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Only the columns this pipeline (raw coils + plasma current, no Thomson) actually
# reads -- parquet lets us skip the rest at the read() call, which matters a lot
# here: each full row also carries Thomson_core/edge Te/ne/R and chord geometry
# that this architecture never touches. Loading those anyway (as the generic
# load_shot_from_hf_row does) roughly doubles per-shot memory for nothing, which
# is what caused a real jetsam OOM kill at 1500 shots even though 900 shots fit
# fine with the full-column loader in train_submission_models.py.
NEEDED_COLUMNS = (
    ["source", "efit_times", "efit_psirz", "magnetics_time", "magnetics_plasma_current_times"]
    + [f"magnetics_{sig}" for sig in D3D_MAGNETICS_SIGNALS]
    + list(EFIT_SCALAR_LABELS)
)


def load_shot_lean(path: Path) -> dict:
    """Like load_shot_from_hf_row, but reads (and therefore loads into memory)
    only NEEDED_COLUMNS -- no Thomson data -- since this pipeline is raw-coil-only."""
    row = pd.read_parquet(path, columns=NEEDED_COLUMNS).iloc[0]
    return load_shot_from_hf_row(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-shots", type=int, default=1500,
                         help="cap the number of local DIII-D shots used (0 = all available)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=128)
    parser.add_argument("--val-frac", type=float, default=0.1,
                         help="fraction of shots (not frames) held out for validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    d3d_files = sorted(D3D_LOCAL_DIR.glob("*.parquet"))
    if args.max_shots:
        d3d_files = d3d_files[:args.max_shots]
    print(f"Training on {len(d3d_files)} local DIII-D shots")

    # Process one shot at a time (lean columns only, no Thomson) so at most one
    # shot's raw+processed data is ever resident -- holding all N raw rows AND all
    # N processed shot dicts simultaneously (the old approach) roughly doubled
    # peak memory and caused a real jetsam OOM kill at 1500 shots.
    t0 = time.time()
    X_parts, Y_parts, S_parts, id_parts = [], [], [], []
    for i, f in enumerate(d3d_files):
        shot = load_shot_lean(f)
        shot["shot_index"] = i
        Xi, Yi, Si, idi = build_feature_matrix([shot], include_thomson=False)
        X_parts.append(Xi); Y_parts.append(Yi); S_parts.append(Si); id_parts.append(idi)
        del shot
        if (i + 1) % 200 == 0:
            print(f"  loaded {i + 1}/{len(d3d_files)} shots ({time.time() - t0:.0f}s)")
    print(f"  Loaded all {len(d3d_files)} shots in {time.time() - t0:.0f}s")

    X = np.concatenate(X_parts, axis=0)
    Y = np.concatenate(Y_parts, axis=0)
    S = np.concatenate(S_parts, axis=0)
    shot_ids = np.concatenate(id_parts, axis=0)
    del X_parts, Y_parts, S_parts, id_parts
    print(f"  X: {X.shape}, Y: {Y.shape}")

    # Split by shot (not frame) so validation frames come from unseen shots.
    rng = np.random.RandomState(args.seed)
    unique_shots = np.unique(shot_ids)
    perm = rng.permutation(unique_shots)
    n_val = max(1, int(len(unique_shots) * args.val_frac))
    val_shots = set(perm[:n_val])
    val_mask = np.isin(shot_ids, list(val_shots))
    train_mask = ~val_mask

    n_features = X.shape[1]
    # float16 storage: halves the resident size of the single biggest array in
    # this script (~8.8GB at full scale in float32) -- this machine's unified
    # memory measurably thrashed into swap at that size during training, not
    # just during the one-time load. Upcast per-batch in the training loop
    # instead, so compute still happens in float32.
    X_train, Y_train = X[train_mask], Y[train_mask].astype(np.float16)
    X_val, Y_val = X[val_mask], Y[val_mask].astype(np.float16)
    print(f"  Train frames: {len(X_train)} ({(~val_mask).sum()}), "
          f"Val frames: {len(X_val)} ({val_mask.sum()}) "
          f"from {len(unique_shots) - n_val}/{n_val} shots")

    # X[mask]/Y[mask] above are copies, not views -- the original X/Y (Y alone
    # is ~8.8GB at full scale) are now redundant with X_train+X_val/Y_train+Y_val
    # combined and were never freed, which measurably contributed to an MPS
    # unified-memory OOM kill at full shot count. Free them before the GPU
    # training loop, which has its own (smaller, but real) memory footprint.
    del X, Y

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    device = resolve_device()
    print(f"  Device: {device}")

    model = FluxDecoderNet(n_features=n_features, base=args.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(Y_train)),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val_s), torch.from_numpy(Y_val)),
        batch_size=args.batch_size,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience, bad_epochs = 12, 0

    for epoch in range(args.epochs):
        model.train()
        train_loss, n_batches = 0.0, 0
        t_epoch = time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(n_batches, 1)

        model.eval()
        val_loss, n_val_batches = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                val_loss += criterion(model(xb), yb).item()
                n_val_batches += 1
        val_loss /= max(n_val_batches, 1)
        scheduler.step(val_loss)

        # MPS's caching allocator draws from the same unified system memory as
        # everything else on Apple Silicon (unlike a discrete GPU's own VRAM),
        # and doesn't release cached blocks back to the OS on its own -- clear
        # it each epoch as a safety net against creeping memory growth.
        if device.type == "mps":
            torch.mps.empty_cache()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch + 1:3d}/{args.epochs}: train={train_loss:.6f} "
              f"val={val_loss:.6f} best={best_val:.6f} lr={lr:.1e} "
              f"({time.time() - t_epoch:.1f}s)")

        if bad_epochs >= patience:
            print(f"  Early stopping at epoch {epoch + 1} (no val improvement for {patience} epochs)")
            break

    model.load_state_dict(best_state)
    model.eval()

    joblib.dump(scaler, MODELS_DIR / "d3d_torch_scaler.joblib")
    torch.save(
        {"state_dict": model.state_dict(), "n_features": n_features, "base": args.base_channels},
        MODELS_DIR / "d3d_torch_model.pt",
    )
    print(f"\nSaved d3d_torch_scaler.joblib and d3d_torch_model.pt to {MODELS_DIR}")
    print(f"Best val MSE: {best_val:.6f}")


if __name__ == "__main__":
    main()
