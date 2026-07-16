from __future__ import annotations

import numpy as np

from fitr.vetting import MISMATCH_SIGMA_THRESHOLD, odd_even_test


def _lightcurve_with_eclipses(
    period, epoch, depths_by_cycle_parity, duration=0.08, noise=1e-4, span=40.0, cadence=0.01, seed=0
):
    """depths_by_cycle_parity: (depth_odd, depth_even)."""
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, span, cadence)
    flux = np.ones_like(time) + rng.normal(0.0, noise, time.size)
    flux_err = np.full_like(time, noise)

    cycle = np.round((time - epoch) / period)
    phase = ((time - epoch) / period) % 1.0
    phase = np.where(phase >= 0.5, phase - 1.0, phase)
    in_transit = np.abs(phase) < duration / 2.0

    depth_odd, depth_even = depths_by_cycle_parity
    odd_mask = in_transit & (np.mod(cycle, 2) != 0)
    even_mask = in_transit & (np.mod(cycle, 2) == 0)
    flux[odd_mask] -= depth_odd
    flux[even_mask] -= depth_even

    return time, flux, flux_err


def test_genuine_planet_no_mismatch():
    time, flux, flux_err = _lightcurve_with_eclipses(
        period=2.5, epoch=1.0, depths_by_cycle_parity=(0.01, 0.01), seed=1
    )
    result = odd_even_test(time, flux, flux_err, period=2.5, epoch=1.0)
    assert result.available
    assert not result.mismatch
    assert result.significance_sigma < MISMATCH_SIGMA_THRESHOLD


def test_half_period_eb_flags_mismatch():
    # A real EB folded at half its true period: odd transits are the
    # primary eclipse (deep), even transits are the secondary (shallow).
    time, flux, flux_err = _lightcurve_with_eclipses(
        period=2.5, epoch=1.0, depths_by_cycle_parity=(0.03, 0.01), seed=2
    )
    result = odd_even_test(time, flux, flux_err, period=2.5, epoch=1.0)
    assert result.available
    assert result.mismatch
    assert result.significance_sigma > MISMATCH_SIGMA_THRESHOLD
    assert result.note is not None
    assert "mismatch" in result.note


def test_insufficient_coverage_is_unavailable():
    # Very short baseline: too few transits/points to split by parity.
    rng = np.random.default_rng(3)
    time = np.arange(0.0, 3.0, 0.01)
    flux = np.ones_like(time) + rng.normal(0.0, 1e-4, time.size)
    flux_err = np.full_like(time, 1e-4)
    result = odd_even_test(time, flux, flux_err, period=2.5, epoch=1.0)
    assert not result.available
    assert result.note is not None


def test_empty_data_is_unavailable():
    result = odd_even_test(np.array([]), np.array([]), np.array([]), period=2.5, epoch=1.0)
    assert not result.available
