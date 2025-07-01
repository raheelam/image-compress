#!/usr/bin/env python
"""Plot rate–distortion results and summary tables.

Reads ``results/metrics.csv`` produced by ``scripts/run_evaluation.py`` and
creates aggregated tables plus vector-PDF figures saved under ``figures/``.

Outputs
-------
1. ``figures/rd_psnr.pdf`` : BPP vs PSNR scatter per dataset
2. ``figures/rd_ssim.pdf`` : BPP vs SSIM scatter per dataset
3. ``figures/summary.csv`` : mean metrics per dataset & method
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RESULTS_CSV = Path("results/metrics.csv")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


def _load_df() -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"{RESULTS_CSV} not found. Run `make eval` first.")

    df = pd.read_csv(RESULTS_CSV)
    return df


def _save_summary(df: pd.DataFrame) -> None:
    summary = (
        df.groupby(["dataset", "method"])
        .agg(bpp=("bpp", "mean"), psnr=("psnr", "mean"), ssim=("ssim", "mean"))
        .reset_index()
    )
    summary_csv = FIG_DIR / "summary.csv"
    summary.to_csv(summary_csv, index=False, float_format="%.3f")
    print(f"[PLOT] Saved summary → {summary_csv}")


# -----------------------------------------------------------------------------
# Rate–distortion plots
# -----------------------------------------------------------------------------


def _plot_rd(df: pd.DataFrame, metric: str, fname: str) -> None:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        data=df,
        x="bpp",
        y=metric,
        hue="method",
        style="dataset",
        palette="Set2",
        s=40,
    )
    plt.title(f"Rate–Distortion ({metric.upper()})")
    plt.xlabel("Bits per pixel (bpp)")
    plt.ylabel(metric.upper())
    plt.tight_layout()
    out_path = FIG_DIR / fname
    plt.savefig(out_path, dpi=300, format="pdf")
    plt.close()
    print(f"[PLOT] Saved figure → {out_path}")


def main() -> None:
    df = _load_df()
    _save_summary(df)

    _plot_rd(df, "psnr", "rd_psnr.pdf")
    _plot_rd(df, "ssim", "rd_ssim.pdf")
    print("[PLOT] All figures generated.")


if __name__ == "__main__":
    main() 