"""Adaptive-k MLP predictor.

Run as a script:
    python -m src.adaptive_k --train        # train model
    python -m src.adaptive_k --predict IMG  # predict k for image
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from PIL import UnidentifiedImageError

from .features import compute_descriptors

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path("models/adaptive_k.pt")


class AdaptiveKNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,6) → (B,1)
        return self.net(x) * 248 + 8  # scale to [8,256]


def _load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


def predict_k(model: AdaptiveKNet, img_path: Path) -> int:
    img = _load_image(img_path)
    feats = compute_descriptors(img)
    x = torch.as_tensor([list(feats.values())], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        k = model(x).round().cpu().item()
    return int(k)


def _gather_image_paths() -> List[Path]:
    roots = [
        Path("data/kodak"),
        Path("data/clic24_val"),
        Path("data/div2k/DIV2K_valid_HR"),
    ]
    paths: List[Path] = []
    for r in roots:
        if r.exists():
            for ext in ("*.png", "*.jpg", "*.bmp", "*.ppm"):
                paths.extend(r.glob(ext))
    return paths


def _label_heuristic(img: np.ndarray) -> int:
    """Heuristic to choose palette size k from colour complexity.

    1. Downsample to 64×64 to reduce unique colour count noise.
    2. Count unique colours.  Map to nearest power-of-two between 8 and 256.
    """
    from skimage.transform import resize

    small = resize(img, (64, 64), order=0, preserve_range=True, anti_aliasing=False)
    unique = len(np.unique(small.reshape(-1, 3), axis=0))
    exp = int(np.clip(np.round(np.log2(unique + 1e-8)), 3, 8))  # 2^3=8 … 2^8=256
    return int(2 ** exp)


def train(args: argparse.Namespace) -> None:
    """Train AdaptiveKNet using self-generated pseudo-labels.

    The procedure is lightweight (<2 min CPU) because it computes descriptors
    once per image (≈150 images in our datasets) and optimises a tiny MLP.
    """

    image_paths = _gather_image_paths()
    if not image_paths:
        raise RuntimeError("No training images found under data/… directories. Did you download the datasets?")

    feats_list: list[list[float]] = []
    labels: list[int] = []

    for p in image_paths:
        try:
            img = _load_image(p)
        except (UnidentifiedImageError, OSError) as e:
            print(f"[WARN] Skipping unreadable image {p.name}: {e}")
            continue
        feats = compute_descriptors(img)
        feats_list.append(list(feats.values()))
        labels.append(_label_heuristic(img))

    X = torch.as_tensor(feats_list, dtype=torch.float32, device=DEVICE)
    y = torch.as_tensor(labels, dtype=torch.float32, device=DEVICE).unsqueeze(1)

    model = AdaptiveKNet().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    n_epochs = 300
    print(f"[TRAIN] Adaptive-k: {len(image_paths)} images, {n_epochs} epochs on {DEVICE}…")
    for epoch in range(1, n_epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        if epoch % 50 == 0 or epoch == n_epochs:
            mae = (pred.detach() - y).abs().mean().item()
            print(f"  epoch {epoch:03d}  loss={loss.item():.2f}  MAE={mae:.2f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[TRAIN] Saved model → {MODEL_PATH}")


def cli(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Adaptive-k predictor")
    p.add_argument("--train", action="store_true", help="train the model")
    p.add_argument("--predict", type=Path, help="image to predict k for")
    args = p.parse_args(argv)

    if args.train:
        train(args)
    elif args.predict:
        model = AdaptiveKNet().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        k = predict_k(model, args.predict)
        print(f"Predicted k = {k}")
    else:
        p.print_help()


if __name__ == "__main__":
    cli() 