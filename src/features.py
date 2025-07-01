"""Feature extraction utilities for adaptive-k predictor.

All feature functions return a scalar or 1D numpy array normalised to [0,1].
"""
from __future__ import annotations

import numpy as np
from skimage import color, filters, feature
from typing import Dict


__all__ = [
    "compute_descriptors",
]


def _entropy(img: np.ndarray) -> float:
    hist, _ = np.histogram(img, bins=256, range=(0, 1), density=True)
    hist += 1e-8  # avoid log0
    return float(-np.sum(hist * np.log2(hist)) / 8.0)  # normalise to [0,1]


def _edge_density(gray: np.ndarray) -> float:
    edges = feature.canny(gray, sigma=1.0)
    return float(edges.mean())


def _dominant_hues(hsv: np.ndarray, k: int = 5) -> float:
    h = hsv[..., 0].reshape(-1, 1)
    # simple 1D k-means++ via scikit-learn
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init="auto", random_state=0)
    km.fit(h)
    # count clusters with >5 % pixels
    counts = np.bincount(km.labels_)
    dominant = (counts / counts.sum() > 0.05).sum()
    return dominant / k  # normalise


def _colour_variance(img: np.ndarray) -> float:
    return float(np.var(img))


def compute_descriptors(img: np.ndarray) -> Dict[str, float]:
    """Compute low-cost descriptors on an RGB image.

    Parameters
    ----------
    img : np.ndarray
        float RGB image in [0,1], shape (H,W,3).

    Returns
    -------
    dict
        Mapping of descriptor name to scalar value in [0,1].
    """
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.0

    gray = color.rgb2gray(img)
    hsv = color.rgb2hsv(img)

    desc = {
        "entropy": _entropy(gray),
        "edge_density": _edge_density(gray),
        "dom_hues": _dominant_hues(hsv),
        "colour_var": _colour_variance(img),
        "mean_sat": float(hsv[..., 1].mean()),
        # resolution class: log2arithmic normalised (256-4096)
        "resolution": np.clip(np.log2(max(img.shape[:2])) / 12.0, 0, 1),
    }
    return desc 