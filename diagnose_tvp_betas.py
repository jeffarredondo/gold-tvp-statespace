"""
Diagnostics on the smoothed beta paths from tvp_gold_model.py.

Two questions this answers visually:

1. Is the USD beta actually noisier than the real-rate beta, or does it
   just look that way on the line chart? Compare the distribution of
   period-to-period *changes* in each beta (scatter over time + a
   quick std-dev comparison). A beta that's genuinely tracking a slow
   regime should have small, low-variance changes; a beta chasing
   observation noise will show large, choppy changes with no structure.

2. Are the two betas trading variance with each other? If real_rate_diff
   and usd_logret are collinear, the Kalman filter can push a given
   period's residual into either beta and still fit the data -- meaning
   a spike in one beta's change coincides with an opposite spike in the
   other. Scatter of delta_beta_real_rate vs delta_beta_usd tests this
   directly: a strong negative correlation there is a smoking gun for
   "the model can't tell which regressor deserves credit," not "both
   coefficients are independently meaningful."
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BETAS_PATH = "tvp_betas.csv"


def main():
    betas = pd.read_csv(BETAS_PATH, index_col=0, parse_dates=True)

    d_real_rate = betas["beta_real_rate"].diff()
    d_usd = betas["beta_usd"].diff()

    print("--- Period-to-period change in each beta ---")
    print(f"  std(delta beta_real_rate): {d_real_rate.std():.5f}")
    print(f"  std(delta beta_usd):       {d_usd.std():.5f}")
    print(f"  ratio (usd / real_rate):   {d_usd.std() / d_real_rate.std():.2f}x")

    corr = d_real_rate.corr(d_usd)
    print(f"\n  corr(delta beta_real_rate, delta beta_usd): {corr:.3f}")
    print("  (a strong negative correlation suggests the filter is trading")
    print("   variance between the two betas rather than each one tracking")
    print("   its own independent signal)")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # top row: scatter of each beta's period-over-period change over time
    axes[0, 0].scatter(betas.index, d_real_rate, s=6, alpha=0.4, color="tab:blue")
    axes[0, 0].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[0, 0].set_title(f"Δ beta_real_rate per period (std={d_real_rate.std():.4f})")

    axes[0, 1].scatter(betas.index, d_usd, s=6, alpha=0.4, color="tab:orange")
    axes[0, 1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[0, 1].set_title(f"Δ beta_usd per period (std={d_usd.std():.4f})")
    # same y-scale as the real-rate panel so the noise difference is honest,
    # not an artifact of matplotlib auto-scaling two very different ranges
    shared_ylim = max(abs(d_real_rate).max(), abs(d_usd).max()) * 1.1
    axes[0, 0].set_ylim(-shared_ylim, shared_ylim)
    axes[0, 1].set_ylim(-shared_ylim, shared_ylim)

    # bottom-left: are the two betas' changes trading off against each other?
    axes[1, 0].scatter(d_real_rate, d_usd, s=8, alpha=0.4, color="tab:green")
    axes[1, 0].axhline(0, color="gray", linewidth=0.6)
    axes[1, 0].axvline(0, color="gray", linewidth=0.6)
    axes[1, 0].set_xlabel("Δ beta_real_rate")
    axes[1, 0].set_ylabel("Δ beta_usd")
    axes[1, 0].set_title(f"Do the betas trade variance? corr = {corr:.3f}")

    # bottom-right: distribution comparison
    axes[1, 1].hist(d_real_rate.dropna(), bins=60, alpha=0.6, label="Δ beta_real_rate", color="tab:blue")
    axes[1, 1].hist(d_usd.dropna(), bins=60, alpha=0.6, label="Δ beta_usd", color="tab:orange")
    axes[1, 1].set_title("Distribution of period-over-period changes")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("tvp_beta_diagnostics.png", dpi=150)
    print("\nSaved plot to tvp_beta_diagnostics.png")


if __name__ == "__main__":
    main()