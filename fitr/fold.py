"""Phase folding and binning of light curves."""

from __future__ import annotations

import numpy as np

AUTO_BIN_THRESHOLD = 20_000
AUTO_BIN_TARGET = 2_000


def fold(time: np.ndarray, period: float, epoch: float) -> np.ndarray:
    """Fold `time` on `period`/`epoch`, returning phase in [-0.5, 0.5)."""
    phase = ((time - epoch) / period) % 1.0
    phase = np.where(phase >= 0.5, phase - 1.0, phase)
    return phase


def bin_folded(
    phase: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse-variance weighted binning of folded data. Empty bins are dropped."""
    edges = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_idx = np.clip(np.digitize(phase, edges) - 1, 0, n_bins - 1)

    weights = 1.0 / np.square(flux_err)
    centers = []
    means = []
    errs = []
    for i in range(n_bins):
        mask = bin_idx == i
        if not np.any(mask):
            continue
        w = weights[mask]
        w_sum = np.sum(w)
        mean = np.sum(flux[mask] * w) / w_sum
        err = np.sqrt(1.0 / w_sum)
        centers.append(0.5 * (edges[i] + edges[i + 1]))
        means.append(mean)
        errs.append(err)

    return np.asarray(centers), np.asarray(means), np.asarray(errs)


def maybe_autobin(
    phase: np.ndarray, flux: np.ndarray, flux_err: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Auto-bin large curves above AUTO_BIN_THRESHOLD points for fitting speed."""
    if len(phase) <= AUTO_BIN_THRESHOLD:
        return phase, flux, flux_err
    return bin_folded(phase, flux, flux_err, AUTO_BIN_TARGET)
