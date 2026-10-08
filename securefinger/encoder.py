"""ResNet18 fingerprint encoder -> 256-D L2-normalized embedding (+ ArcFace head for training)."""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from . import EMBED_DIM, IMG_SIZE


class FingerprintEncoder(nn.Module):
    """Single-channel ResNet18 backbone with a 256-D projection, output is unit-norm."""

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        net = resnet18(weights=None)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        net.fc = nn.Identity()
        self.backbone = net
        self.proj = nn.Sequential(nn.Linear(512, embed_dim), nn.BatchNorm1d(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self.backbone(x)), dim=1)


class ArcFaceHead(nn.Module):
    """Additive angular margin loss (ArcFace), s=64, m=0.5 by default."""

    def __init__(self, embed_dim: int, n_classes: int, s: float = 64.0, m: float = 0.5):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_classes, embed_dim))
        nn.init.xavier_uniform_(self.W)
        self.s, self.m = s, m
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cos = F.linear(emb, F.normalize(self.W)).clamp(-1 + 1e-7, 1 - 1e-7)
        sin = torch.sqrt(1.0 - cos ** 2)
        phi = cos * self.cos_m - sin * self.sin_m
        phi = torch.where(cos > self.th, phi, cos - self.mm)
        onehot = F.one_hot(labels, cos.size(1)).float()
        logits = self.s * (onehot * phi + (1 - onehot) * cos)
        return F.cross_entropy(logits, labels)


class Embedder:
    """Inference wrapper: preprocessed (1,96,96) array -> 256-D float32 unit vector."""

    def __init__(self, weights: str | None = None, device: str = "cpu"):
        self.device = device
        self.model = FingerprintEncoder().to(device).eval()
        self.trained = False
        self.threshold = None
        if weights and Path(weights).exists():
            state = torch.load(weights, map_location=device)
            self.model.load_state_dict(state["encoder"] if "encoder" in state else state)
            self.threshold = state.get("threshold") if isinstance(state, dict) else None
            self.trained = True

    @torch.no_grad()
    def embed(self, x: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        return self.model(t)[0].cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def embed_batch(self, xs: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(xs).float().to(self.device)
        return self.model(t).cpu().numpy().astype(np.float64)


def export_onnx(model: FingerprintEncoder, path: str) -> None:
    model = model.cpu().eval()
    dummy = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(model, dummy, path, input_names=["image"], output_names=["embedding"],
                      dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
                      opset_version=17)
