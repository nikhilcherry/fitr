from __future__ import annotations

import numpy as np

from fitr.fold import AUTO_BIN_THRESHOLD, bin_folded, fold, maybe_autobin


def test_fold_range():
    time = np.linspace(0, 100, 5000)
    phase = fold(time, period=3.7, epoch=1.2)
    assert np.all(phase >= -0.5)
    assert np.all(phase < 0.5)


def test_fold_centers_transit_at_zero():
    period, epoch = 5.0, 2.0
    time = np.array([epoch, epoch + period, epoch - period])
    phase = fold(time, period, epoch)
    np.testing.assert_allclose(phase, 0.0, atol=1e-9)


def test_bin_folded_shapes_and_weighting():
    rng = np.random.default_rng(0)
    phase = rng.uniform(-0.5, 0.5, 1000)
    flux = np.ones(1000)
    flux_err = np.full(1000, 0.01)

    centers, means, errs = bin_folded(phase, flux, flux_err, n_bins=20)
    assert len(centers) == len(means) == len(errs)
    assert len(centers) <= 20
    np.testing.assert_allclose(means, 1.0, atol=1e-9)


def test_bin_folded_drops_empty_bins():
    phase = np.array([-0.4, -0.4, 0.4])
    flux = np.array([1.0, 1.0, 1.0])
    flux_err = np.array([0.1, 0.1, 0.1])

    centers, means, errs = bin_folded(phase, flux, flux_err, n_bins=10)
    assert len(centers) == 2  # only two non-empty bins


def test_maybe_autobin_passthrough_for_small_curves():
    phase = np.linspace(-0.5, 0.5, 100, endpoint=False)
    flux = np.ones(100)
    flux_err = np.full(100, 0.01)

    out_phase, out_flux, out_err = maybe_autobin(phase, flux, flux_err)
    np.testing.assert_array_equal(out_phase, phase)


def test_maybe_autobin_bins_large_curves():
    n = AUTO_BIN_THRESHOLD + 1000
    phase = np.linspace(-0.5, 0.5, n, endpoint=False)
    flux = np.ones(n)
    flux_err = np.full(n, 0.01)

    out_phase, out_flux, out_err = maybe_autobin(phase, flux, flux_err)
    assert len(out_phase) < n
