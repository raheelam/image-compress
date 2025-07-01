#!/usr/bin/env python
"""Run baselines and our adaptive compressor on all datasets.

Creates results CSV with columns:
    dataset, image, method, orig_bytes, comp_bytes, bpp, psnr, ssim

Baselines
---------
1. png       : original PNG saved with Pillow (reference for size/quality)
2. k256      : 256-colour k-means palette (no refinement)
3. adaptive  : our full pipeline (adaptive-k + refine)

This script intentionally avoids heavy dependencies like imagecodecs; it uses
Pillow for PNG saving and our own functions for compression.
"""
from __future__ import annotations

# Ensure project root is on sys.path before importing project modules
import sys
from pathlib import Path
import os  # Needed for env vars & cpu count
import argparse

# ---------------------------------------------------------------------------
# Limit BLAS/OpenMP thread fan-out to avoid thermal shutdowns on laptops.
# This must be done BEFORE importing numpy/scipy/sklearn.
# ---------------------------------------------------------------------------
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import csv
import multiprocessing as mp
from pathlib import Path as _P  # renamed to avoid clash
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.compress import compress_image
from src.metrics import psnr_hvs, ssim_index

DATASETS = {
    # Kodak download yielded empty stub files – skip for evaluation to avoid PIL errors
    "clic24": Path("data/clic24_val"),
    "div2k": Path("data/div2k/DIV2K_valid_HR"),
}

RESULTS_DIR = Path("results")
COMP_DIR = RESULTS_DIR / "compressed"
COMP_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = RESULTS_DIR / "metrics.csv"

# ---------------------------
# Argument parsing utilities
# ---------------------------

def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for evaluation options."""
    parser = argparse.ArgumentParser(description="Run image-compression evaluation.")
    default_proc = max(os.cpu_count() // 2, 1)
    parser.add_argument(
        "--processes",
        type=int,
        default=default_proc,
        help=f"Number of worker processes to spawn (default: {default_proc}).",
    )
    return parser.parse_args()


def _safe_open(img_path: Path) -> np.ndarray | None:
    try:
        img = Image.open(img_path).convert("RGB")
        return np.asarray(img).astype(np.float32) / 255.0
    except Exception:
        return None


def _PNG_save(arr: np.ndarray, out_path: Path) -> None:
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img.save(out_path, format="PNG", optimize=True)


def _kmeans256(arr: np.ndarray, out_path: Path) -> None:
    from sklearn.cluster import MiniBatchKMeans as KMeans  # noqa: N812

    h, w, _ = arr.shape
    small_flat = arr[::2, ::2].reshape(-1, 3)  # quarter pixel count ⇒ 4× less RAM
    km = KMeans(n_clusters=256, batch_size=8192, random_state=0).fit(small_flat)

    # Predict labels for full-resolution image using the learned centroids
    flat_full = arr.reshape(-1, 3)
    labels = km.predict(flat_full).astype(np.uint8)
    centroids = km.cluster_centers_
    pal_img = Image.fromarray(labels.reshape(h, w), mode="P")
    palette = (centroids * 255).astype(np.uint8).flatten().tolist()
    pal_img.putpalette(palette + [0] * (768 - len(palette)))
    pal_img.save(out_path, format="PNG", optimize=True, bits=8)


def _process_one(args: Tuple[str, Path]) -> List[Tuple]:
    dataset, img_path = args
    arr = _safe_open(img_path)
    if arr is None:
        return []

    orig_png = COMP_DIR / dataset / "png" / img_path.name
    k256_png = COMP_DIR / dataset / "k256" / img_path.name
    ada_png = COMP_DIR / dataset / "adaptive" / img_path.name

    # ensure dirs
    for p in [orig_png, k256_png, ada_png]:
        p.parent.mkdir(parents=True, exist_ok=True)

    # save original (may already exist)
    if not orig_png.exists():
        _PNG_save(arr, orig_png)

    if not k256_png.exists():
        _kmeans256(arr, k256_png)

    if not ada_png.exists():
        compress_image(img_path, ada_png)

    # metrics
    orig_bytes = orig_png.stat().st_size
    rows: List[Tuple] = []
    for method, pth in [
        ("k256", k256_png),
        ("adaptive", ada_png),
    ]:
        comp_bytes = pth.stat().st_size
        comp = _safe_open(pth)
        if comp is None:
            continue
        psnr_v = psnr_hvs(comp, arr)
        ssim_v = ssim_index(comp, arr)
        bpp = comp_bytes * 8 / (arr.shape[0] * arr.shape[1])
        rows.append((dataset, img_path.name, method, orig_bytes, comp_bytes, bpp, psnr_v, ssim_v))
    return rows


# -------------------------
# Main entry point
# -------------------------

def main() -> None:
    args = _parse_args()

    tasks: List[Tuple[str, Path]] = []
    for ds, root in DATASETS.items():
        if not root.exists():
            continue
        tasks.extend((ds, p) for p in root.glob("*.png"))

    print(
        f"[EVAL] {len(tasks)} images across {len(DATASETS)} datasets – using {args.processes} process(es)…"
    )

    with mp.Pool(processes=args.processes) as pool, open(CSV_PATH, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow("dataset image method orig_bytes comp_bytes bpp psnr ssim".split())
        for rows in tqdm(pool.imap_unordered(_process_one, tasks), total=len(tasks)):
            for r in rows:
                writer.writerow(r)
    print(f"[EVAL] Done. Metrics saved to {CSV_PATH}")


if __name__ == "__main__":
    main() 