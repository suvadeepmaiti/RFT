#!/usr/bin/env python3
"""
rft_vs_empirical_summary.py

Simulate 1D Gaussian random fields, compute empirical cluster counts and sizes,
and compare to Random Field Theory (RFT) theoretical predictions.
Saves results to 'rft_vs_empirical_summary.csv'.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm

# --- Helper functions ---

def generate_field(n, fwhm):
    """
    Generate a normalized 1D Gaussian random field of length n,
    smoothed by a Gaussian kernel of width FWHM.
    """
    x = np.random.randn(n)
    if fwhm > 0:
        sigma = fwhm / 2.3548  
        x = gaussian_filter1d(x, sigma, mode='reflect')
    return (x - x.mean()) / x.std()

def count_clusters(data, thr):
    """
    Count the number of contiguous suprathreshold clusters in 1D data.
    """
    mask = data > thr
    if not mask.any():
        return 0
    idx = np.where(mask)[0]
    # clusters = runs of consecutive indices
    return int(np.sum(np.diff(idx) > 1) + 1)

# --- Main analysis ---

def main():
    # Parameters
    N            = 1000                 # length of each 1D field (voxels)
    n_iter       = 1000                 # number of simulations per FWHM
    fwhm_levels  = [1, 10, 20]          # smoothness levels (1≈very low)
    thresholds   = [1.64, 2.33, 3.09]    # Z-thresholds
    S            = N                    # field length for RFT
    D            = 1                    # dimensionality

    rows = []

    for fwhm in fwhm_levels:
        diffs = []
        for _ in range(n_iter):
            fld = generate_field(N, fwhm)
            diffs.append(np.mean(np.diff(fld)**2))
        W = np.sqrt(np.mean(diffs) / (4 * np.log(2)))

        for u in thresholds:
            counts = [count_clusters(generate_field(N, fwhm), u)
                      for _ in range(n_iter)]
            emp_mean_clusters = np.mean(counts)

            # Theoretical expected suprathreshold voxels E[N]
            EN = (1 - norm.cdf(u)) * S

            # Theoretical expected clusters E[m] via 1D EC formula
            Em = (S * (2 * np.pi)**(-(D + 1) / 2) *
                  W**(-D) * u**(D - 1) * np.exp(-u**2 / 2))

            # Theoretical mean cluster size E[n] = E[N] / E[m]
            En = EN / Em if Em > 0 else np.nan

            rows.append({
                'FWHM': fwhm,
                'Z_threshold': u,
                'W_est': W,
                'Empirical_Mean_Clusters': emp_mean_clusters,
                'Theoretical_EN_voxels': EN,
                'Theoretical_Em_clusters': Em,
                'Theoretical_En_size': En
            })

    df = pd.DataFrame(rows)

    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', None)
    print(df)

    output_csv = 'rft_vs_empirical_summary.csv'
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to '{output_csv}'")

if __name__ == "__main__":
    main()
