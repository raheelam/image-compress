"""High-level compressor combining adaptive-k selection, k-means palette, centroid refinement, and PNG-8 writing."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans

from .adaptive_k import AdaptiveKNet, predict_k, DEVICE, MODEL_PATH as AK_PATH
from .refine_centroid import RefineNet, MODEL_PATH as RC_PATH


def _load_models() -> Tuple[AdaptiveKNet, RefineNet]:
    ak = AdaptiveKNet().to(DEVICE)
    ak.load_state_dict(torch.load(AK_PATH, map_location=DEVICE))
    rc = RefineNet().to(DEVICE)
    rc.load_state_dict(torch.load(RC_PATH, map_location=DEVICE))
    ak.eval(); rc.eval()
    return ak, rc


def compress_image(in_path: Path, out_path: Path) -> None:
    img = Image.open(in_path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0

    ak, rc = _load_models()
    k = predict_k(ak, in_path)

    # run k-means on downsampled copy
    small = arr[::2, ::2].reshape(-1, 3)
    km = KMeans(n_clusters=k, n_init="auto", random_state=0)
    km.fit(small)
    centroids = km.cluster_centers_.astype(np.float32)  # (k,3)

    # refine
    c = torch.as_tensor(centroids.T[None], device=DEVICE)  # (1,3,k)
    with torch.no_grad():
        c_ref = rc(c).squeeze(0).T.cpu().numpy()

    # assign full-res pixels
    flat = arr.reshape(-1, 3)
    labels = np.argmin(((flat[:, None, :] - c_ref[None]) ** 2).sum(-1), axis=1).astype(np.uint8)
    pal_img = Image.fromarray(labels.reshape(arr.shape[:2]), mode="P")
    palette = (c_ref * 255).astype(np.uint8).flatten().tolist()
    pal_img.putpalette(palette + [0] * (768 - len(palette)))
    pal_img.save(out_path, format="PNG", optimize=True, bits=8)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Compress an image to PNG-8 using adaptive palette")
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()

    compress_image(args.input, args.output) 