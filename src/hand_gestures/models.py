"""Gesture models: two static classifiers (MLP on hand-crafted features, graph network on the
skeleton) and one causal temporal network for dynamic gestures."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from hand_gestures.landmarks import EDGES


class MLPClassifier(nn.Module):
    """Input: (B, 67) features from ``landmarks.features``."""

    def __init__(
        self, num_classes: int, in_dim: int = 67, hidden: int = 256, depth: int = 3, dropout: float = 0.2
    ):
        super().__init__()
        layers, dim = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
            dim = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


def skeleton_adjacency() -> torch.Tensor:
    """Symmetrically normalised adjacency with self-loops, D^-1/2 (A + I) D^-1/2."""
    a = np.eye(21, dtype=np.float32)
    for i, j in EDGES:
        a[i, j] = a[j, i] = 1.0
    d = 1.0 / np.sqrt(a.sum(1))
    return torch.from_numpy(a * d[:, None] * d[None, :])


class GraphConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.linear = nn.Linear(cin, cout)
        self.norm = nn.LayerNorm(cout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(adjacency @ self.linear(x)))


class HandGCN(nn.Module):
    """Graph convolutions along the bones of the hand. Input: (B, 21, 2) normalised points; a learned
    per-joint embedding tells the network which joint is which."""

    def __init__(self, num_classes: int, widths: tuple[int, ...] = (64, 128, 128), dropout: float = 0.2):
        super().__init__()
        self.register_buffer("adjacency", skeleton_adjacency())
        self.joint_embedding = nn.Parameter(torch.zeros(21, widths[0]))
        self.input = nn.Linear(2, widths[0])
        self.layers = nn.ModuleList(GraphConv(a, b) for a, b in zip(widths, widths[1:], strict=False))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * widths[-1], num_classes))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        x = self.input(points) + self.joint_embedding
        for layer in self.layers:
            x = (
                x + layer(x, self.adjacency)
                if x.shape[-1] == layer.linear.out_features
                else layer(x, self.adjacency)
            )
        return self.head(torch.cat([x.mean(1), x.amax(1)], dim=-1))


class TimestepNorm(nn.Module):
    """LayerNorm over channels at each time step. GroupNorm/BatchNorm statistics would mix in
    future frames and break causality."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class CausalConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel, dilation=dilation)
        self.norm = TimestepNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(F.pad(x, (self.pad, 0)))  # pad only on the left: no look into the future
        return x + self.dropout(F.gelu(self.norm(y)))


class GestureTCN(nn.Module):
    """Causal temporal convolutional network over per-frame features (B, T, F) -> logits (B, C).
    Dilations 1-2-4-8 give a receptive field of 31 frames (~1 s at 30 fps), and causality means the
    same weights work on a live stream."""

    def __init__(
        self, num_classes: int, in_dim: int, channels: int = 128, dilations: tuple[int, ...] = (1, 2, 4, 8)
    ):
        super().__init__()
        self.input = nn.Conv1d(in_dim, channels, 1)
        self.blocks = nn.Sequential(*(CausalConvBlock(channels, d) for d in dilations))
        self.head = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.input(x.transpose(1, 2)))
        return self.head(h[:, :, -1])  # the last step has seen the whole window


class GestureGRU(nn.Module):
    """Recurrent alternative, for comparison."""

    def __init__(self, num_classes: int, in_dim: int, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, layers, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1])


def build_static(name: str, num_classes: int) -> nn.Module:
    return {"mlp": MLPClassifier, "gcn": HandGCN}[name](num_classes)


def build_temporal(name: str, num_classes: int, in_dim: int) -> nn.Module:
    return {"tcn": GestureTCN, "gru": GestureGRU}[name](num_classes, in_dim)
