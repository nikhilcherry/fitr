from __future__ import annotations

import numpy as np

from fitr.fit import fit_all, fit_model
from fitr.fold import fold, maybe_autobin
from fitr.models import ALL_MODELS


def test_planet_recovery_and_param_tolerance(planet_curve):
    results = fit_all(planet_curve.lc, planet_curve.period, planet_curve.epoch)
    best = min(results, key=lambda r: r.bic)
    assert best.model_name == "planet"

    rp_rs_true = planet_curve.true_params["rp_rs"]
    rp_rs_fit = best.params["rp_rs"]
    assert abs(rp_rs_fit - rp_rs_true) / rp_rs_true < 0.20


def test_eb_recovery(eb_curve):
    results = fit_all(eb_curve.lc, eb_curve.period, eb_curve.epoch)
    best = min(results, key=lambda r: r.bic)
    assert best.model_name == "eb"


def test_starspot_recovery(starspot_curve):
    results = fit_all(starspot_curve.lc, starspot_curve.period, starspot_curve.epoch)
    best = min(results, key=lambda r: r.bic)
    assert best.model_name == "starspot"


def test_blend_recovery_allows_ambiguous_with_planet(blend_curve):
    from fitr.compare import compare

    results = fit_all(blend_curve.lc, blend_curve.period, blend_curve.epoch)
    phase = fold(blend_curve.lc.time, blend_curve.period, blend_curve.epoch)
    phase, flux, flux_err = maybe_autobin(phase, blend_curve.lc.flux, blend_curve.lc.flux_err)
    comparison = compare(results, phase, flux, flux_err)

    if comparison.verdict == "ambiguous":
        assert "blend" in comparison.tied_models
    else:
        assert comparison.winner == "blend"


def test_failed_fit_does_not_crash(monkeypatch):
    model = ALL_MODELS["planet"]

    def broken_evaluate(self, phase, params, period):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(type(model), "evaluate", broken_evaluate)

    phase = np.linspace(-0.5, 0.5, 100, endpoint=False)
    flux = np.ones(100)
    flux_err = np.full(100, 0.01)

    result = fit_model(model, phase, flux, flux_err, period=3.0)
    assert result.converged is False
    assert result.chi2 == float("inf")


def test_null_curve_yields_no_significant_signal(null_curve):
    from fitr.compare import compare

    results = fit_all(null_curve.lc, null_curve.period, null_curve.epoch)
    phase = fold(null_curve.lc.time, null_curve.period, null_curve.epoch)
    phase, flux, flux_err = maybe_autobin(phase, null_curve.lc.flux, null_curve.lc.flux_err)
    comparison = compare(results, phase, flux, flux_err)

    assert comparison.verdict == "no_significant_signal"
    assert comparison.winner is None


def test_determinism_byte_identical_params():
    from tests.conftest import make_planet_curve

    sc1 = make_planet_curve()
    sc2 = make_planet_curve()

    results1 = fit_all(sc1.lc, sc1.period, sc1.epoch)
    results2 = fit_all(sc2.lc, sc2.period, sc2.epoch)

    for r1, r2 in zip(results1, results2):
        assert r1.model_name == r2.model_name
        assert r1.chi2 == r2.chi2
        assert r1.params == r2.params
