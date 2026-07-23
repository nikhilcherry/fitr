from __future__ import annotations

import numpy as np
import pytest

from fitr.models._util import estimate_depth_and_duration


def test_duration_uses_fixed_bin_width_even_with_empty_bins():
    # Two populated bins with several empty bins between them on the
    # underlying fixed edge grid: bin 0 (center -0.45) and bin 5 (center
    # 0.05) out of 10 bins each 0.1 wide. centers[1] - centers[0] between
    # those two *populated* bins is 0.5 -- 5x the true 0.1 bin width -- so
    # deriving duration from that gap (the old behavior) would badly
    # overestimate it whenever intervening bins are empty.
    n_bins = 10
    bin_width = 1.0 / n_bins
    phase = np.array([-0.45, 0.05])
    flux = np.array([1.0, 0.9])  # bin 5 is the deep one

    depth, duration, t0_est = estimate_depth_and_duration(phase, flux, n_bins=n_bins)

    assert depth == pytest.approx(0.05, abs=1e-9)
    assert t0_est == pytest.approx(0.05, abs=1e-9)
    # Exactly one bin (bin 5) is below half-depth, so duration is one true
    # bin width -- not the 0.5 the old centers-gap formula would give.
    assert duration == pytest.approx(bin_width, abs=1e-9)


def test_duration_matches_bin_width_times_count_below_half_depth():
    # All 50 bins populated (no gaps): duration should equal exactly
    # (number of bins below half-depth) * (1 / n_bins).
    n_bins = 50
    phase = np.linspace(-0.5, 0.5, n_bins, endpoint=False) + 0.5 / n_bins
    flux = np.ones(n_bins)
    in_transit = np.abs(phase) < 3 * (1.0 / n_bins)
    flux[in_transit] = 0.9

    depth, duration, _ = estimate_depth_and_duration(phase, flux, n_bins=n_bins)

    below_half_depth_count = int(np.sum(flux < 1.0 - 0.5 * depth))
    assert duration == pytest.approx(below_half_depth_count / n_bins, abs=1e-9)
