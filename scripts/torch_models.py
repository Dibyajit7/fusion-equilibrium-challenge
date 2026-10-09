"""
Conv-decoder architecture for DIII-D flux-map prediction, shared between the
training script (scripts/train_torch_d3d.py) and inference (submission_skeleton.py).

Unlike the current sklearn pipeline (StandardScaler -> PCA(50) -> MLPRegressor,
which regresses 50 linear PCA coefficients), this predicts the 65x65 grid directly
through a convolutional decoder. The PCA step was already capturing 100% of the
training variance at 50 components, so the sklearn pipeline's error comes from the
MLP regression itself, not the compression -- the motivation here is a more
expressive nonlinear regressor (depth + spatial conv inductive bias) trained longer
with a GPU, not a bigger target basis.

F.interpolate(..., size=(H, W)) is used between conv blocks instead of
ConvTranspose2d so each stage's exact output size is explicit (65 is not a power of
2 and doesn't fall out of stride-2 transposed convolutions cleanly).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

GRID = 65


class FluxDecoderNet(nn.Module):
    def __init__(self, n_features: int, base: int = 128, dropout: float = 0.1):
        super().__init__()
        self.base = base
        self.fc = nn.Sequential(
            nn.Linear(n_features, 256), nn.ReLU(inplace=True), nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 512), nn.ReLU(inplace=True), nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, base * 5 * 5), nn.ReLU(inplace=True),
        )

        def conv_block(cin: int, cout: int) -> nn.Module:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            )

        self.block1 = conv_block(base, base // 2)       # applied at 9x9
        self.block2 = conv_block(base // 2, base // 4)   # applied at 17x17
        self.block3 = conv_block(base // 4, base // 8)   # applied at 33x33
        self.block4 = conv_block(base // 8, base // 8)   # applied at 65x65 (final resolution)
        self.out_conv = nn.Conv2d(base // 8, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.fc(x).view(-1, self.base, 5, 5)
        z = F.interpolate(z, size=(9, 9), mode="bilinear", align_corners=False)
        z = self.block1(z)
        z = F.interpolate(z, size=(17, 17), mode="bilinear", align_corners=False)
        z = self.block2(z)
        z = F.interpolate(z, size=(33, 33), mode="bilinear", align_corners=False)
        z = self.block3(z)
        z = F.interpolate(z, size=(GRID, GRID), mode="bilinear", align_corners=False)
        z = self.block4(z)
        return self.out_conv(z).squeeze(1)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
