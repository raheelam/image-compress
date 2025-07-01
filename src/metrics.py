"""Metric wrappers for SSIM, PSNR-HVS, and ΔE2000."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage import color as _color
from colormath.color_diff import delta_e_cie2000


def psnr_hvs(img: np.ndarray, ref: np.ndarray) -> float:  # placeholder
    # For brevity, reuse PSNR; real HVS variant can be added later
    return psnr(ref, img, data_range=1.0)


def ssim_index(img: np.ndarray, ref: np.ndarray) -> float:
    return ssim(ref, img, channel_axis=-1, data_range=1.0)


def delta_e2000(img: np.ndarray, ref: np.ndarray) -> float:
    lab1 = _color.rgb2lab(img)
    lab2 = _color.rgb2lab(ref)
    return float(delta_e_cie2000(lab1.reshape(-1, 3), lab2.reshape(-1, 3)).mean()) 