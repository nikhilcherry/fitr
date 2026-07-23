"""Shared helpers for initial-guess heuristics across models."""

from __future__ import annotations

import numpy as np


def coarse_bin(
    phase: np.ndarray, flux: np.ndarray, n_bins: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(-0.5, 0.5, n_bins + 1)
    idx = np.clip(np.digitize(phase, edges) - 1, 0, n_bins - 1)
    centers = []
    means = []
    for i in range(n_bins):
        mask = idx == i
        if not np.any(mask):
            continue
        centers.append(0.5 * (edges[i] + edges[i + 1]))
        means.append(np.mean(flux[mask]))
    return np.asarray(centers), np.asarray(means)


def estimate_depth_and_duration(
    phase: np.ndarray, flux: np.ndarray, n_bins: int = 50
) -> tuple[float, float, float]:
    """Rough (depth, duration_phase, phase_of_minimum) from a coarse binned curve."""
    centers, means = coarse_bin(phase, flux, n_bins=n_bins)
    if len(means) == 0:
        return 0.01, 0.05, 0.0

    baseline = np.median(means)
    min_idx = np.argmin(means)
    depth = max(baseline - means[min_idx], 1e-6)
    t0_est = centers[min_idx]

    # Fixed bin width from the (always evenly spaced) edge grid, not from
    # adjacent *populated* bin centers -- centers[1] - centers[0] silently
    # overestimates the spacing (and thus the duration) whenever a bin
    # between them is empty and gets dropped by coarse_bin.
    bin_width = 1.0 / n_bins
    half_depth = baseline - 0.5 * depth
    below = means < half_depth
    duration = np.sum(below) * bin_width
    duration = max(duration, 0.005)

    return float(depth), float(duration), float(t0_est)


def a_rs_from_duration(duration_phase: float, inc_deg: float = 89.0) -> float:
    """Rough scaled semi-major axis from a fractional transit duration."""
    duration_phase = max(duration_phase, 1e-3)
    inc = np.deg2rad(inc_deg)
    a_rs = 1.0 / (np.pi * duration_phase * np.sin(inc))
    return float(np.clip(a_rs, 2.0, 200.0))
