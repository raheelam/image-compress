"""Centroid refinement network.

Takes initial palette centroids and a downsample of the image, outputs refined centroids.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path("models/refine_centroid.pt")


class RefineNet(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.conv1 = nn.Conv1d(3, hidden, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden, 3, kernel_size=1)
        self.act = nn.GELU()
        self.tanh = nn.Tanh()

    def forward(self, c: torch.Tensor) -> torch.Tensor:  # (B,3,k)
        y = self.act(self.conv1(c))
        delta = self.tanh(self.conv2(y)) * 0.05
        return torch.clamp(c + delta, 0.0, 1.0)


def train(args: argparse.Namespace) -> None:
    """Quickly train RefineNet to approximate identity mapping.

    This lightweight routine samples random palettes and teaches the network
    to output virtually the same centroids (small residual).  It suffices to
    initialise weights so that later fine-tuning can start from near-identity.
    Runtime <30 s on CPU.
    """

    model = RefineNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    n_batches = 500
    for i in range(1, n_batches + 1):
        # sample random k (power of two between 8 and 256)
        exp = torch.randint(3, 9, (1,)).item()  # 2^3=8 .. 2^8=256
        k = 2 ** exp
        c = torch.rand((1, 3, k), device=DEVICE)

        pred = model(c)
        loss = loss_fn(pred, c)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 100 == 0 or i == n_batches:
            print(f"  batch {i}/{n_batches}  loss={loss.item():.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[TRAIN] Saved RefineNet → {MODEL_PATH}")


def cli(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Centroid refinement net")
    p.add_argument("--train", action="store_true")
    args = p.parse_args(argv)

    if args.train:
        train(args)
    else:
        p.print_help()


if __name__ == "__main__":
    cli() 