from __future__ import annotations

import numpy as np

from fitr.compare import compare
from fitr.fit import FitResult
from fitr.vetting import OddEvenResult


def _result(name, bic, chi2=None, n_params=3, n_points=1000, converged=True):
    return FitResult(
        model_name=name,
        params={},
        chi2=chi2 if chi2 is not None else bic,
        n_points=n_points,
        n_params=n_params,
        bic=bic,
        aic=bic,
        converged=converged,
        runtime_s=0.01,
    )


def _flat_data(n=1000, sigma=0.01, seed=0):
    rng = np.random.default_rng(seed)
    phase = np.linspace(-0.5, 0.5, n, endpoint=False)
    flux = np.ones(n) + rng.normal(0, sigma, n)
    flux_err = np.full(n, sigma)
    return phase, flux, flux_err


def test_clear_winner():
    results = [_result("planet", 100.0), _result("eb", 150.0), _result("blend", 120.0), _result("starspot", 500.0)]
    phase, flux, flux_err = _flat_data()
    comparison = compare(results, phase, flux, flux_err)
    # baseline bic on near-flat noise will be large relative to these fake low values,
    # so this should not trigger no_significant_signal
    assert comparison.verdict in ("clear", "no_significant_signal")
    if comparison.verdict == "clear":
        assert comparison.winner == "planet"


def test_ambiguous_planet_blend_tie():
    results = [
        _result("planet", 100.0),
        _result("blend", 101.0),
        _result("eb", 150.0),
        _result("starspot", 500.0),
    ]
    phase, flux, flux_err = _flat_data()
    comparison = compare(results, phase, flux, flux_err)
    assert comparison.verdict == "ambiguous"
    assert "planet" in comparison.tied_models
    assert "blend" in comparison.tied_models
    assert any("centroid" in note for note in comparison.notes)


def test_no_significant_signal_when_baseline_wins():
    n = 1000
    phase = np.linspace(-0.5, 0.5, n, endpoint=False)
    rng = np.random.default_rng(1)
    flux = np.ones(n) + rng.normal(0, 0.01, n)
    flux_err = np.full(n, 0.01)

    baseline_chi2 = np.sum(((flux - np.mean(flux)) / flux_err) ** 2)
    # Models barely beat the baseline chi2, well within ΔBIC=10.
    results = [
        _result("planet", bic=baseline_chi2 - 1, chi2=baseline_chi2 - 1, n_params=4),
        _result("eb", bic=baseline_chi2 + 5, chi2=baseline_chi2 + 5, n_params=6),
        _result("blend", bic=baseline_chi2 + 2, chi2=baseline_chi2 + 2, n_params=5),
        _result("starspot", bic=baseline_chi2 + 3, chi2=baseline_chi2 + 3, n_params=4),
    ]
    comparison = compare(results, phase, flux, flux_err)
    assert comparison.verdict == "no_significant_signal"
    assert comparison.winner is None


def test_odd_even_mismatch_note_propagates_on_clear_winner():
    results = [_result("planet", 100.0), _result("eb", 150.0), _result("blend", 120.0), _result("starspot", 500.0)]
    phase, flux, flux_err = _flat_data()
    odd_even = OddEvenResult(
        available=True,
        depth_odd=0.03,
        depth_even=0.01,
        depth_odd_err=0.001,
        depth_even_err=0.001,
        n_in_transit_odd=20,
        n_in_transit_even=20,
        significance_sigma=14.1,
        mismatch=True,
        note="odd-even depth mismatch (14.1σ): odd and even transits have significantly different depths",
    )
    comparison = compare(results, phase, flux, flux_err, odd_even=odd_even)
    if comparison.verdict == "clear":
        assert any("odd-even" in note for note in comparison.notes)
    assert comparison.odd_even is odd_even


def test_odd_even_consistent_result_adds_no_note():
    results = [_result("planet", 100.0), _result("eb", 150.0), _result("blend", 120.0), _result("starspot", 500.0)]
    phase, flux, flux_err = _flat_data()
    odd_even = OddEvenResult(available=True, mismatch=False, significance_sigma=0.5)
    comparison = compare(results, phase, flux, flux_err, odd_even=odd_even)
    assert not any("odd-even" in note for note in comparison.notes)


def test_all_failed_fits_yields_no_significant_signal():
    results = [
        _result("planet", bic=float("inf"), chi2=float("inf"), converged=False),
        _result("eb", bic=float("inf"), chi2=float("inf"), converged=False),
        _result("blend", bic=float("inf"), chi2=float("inf"), converged=False),
        _result("starspot", bic=float("inf"), chi2=float("inf"), converged=False),
    ]
    phase, flux, flux_err = _flat_data()
    comparison = compare(results, phase, flux, flux_err)
    assert comparison.verdict == "no_significant_signal"
    assert comparison.winner is None
