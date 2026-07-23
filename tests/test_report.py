from __future__ import annotations

import json

import numpy as np

from fitr.compare import compare
from fitr.fit import FitResult
from fitr.report import to_json, to_text
from fitr.vetting import OddEvenResult


def _result(name, bic, chi2=None, n_params=3, n_points=1000, converged=True):
    return FitResult(
        model_name=name,
        params={"a": 1.23456789},
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


def _clear_comparison(odd_even=None):
    results = [_result("planet", 100.0), _result("eb", 150.0), _result("blend", 300.0)]
    phase, flux, flux_err = _flat_data()
    return compare(results, phase, flux, flux_err, odd_even=odd_even)


def test_to_json_roundtrips_and_has_expected_keys():
    comparison = _clear_comparison()
    payload = json.loads(to_json(comparison))
    assert payload["verdict"] == comparison.verdict
    assert payload["odd_even"] is None
    assert len(payload["models"]) == 3
    assert payload["models"][0]["model"] in {"planet", "eb", "blend"}


def test_to_text_mentions_clear_winner():
    comparison = _clear_comparison()
    text = to_text(comparison)
    if comparison.verdict == "clear":
        assert f"clear winner = {comparison.winner}" in text


def test_to_text_and_json_include_available_odd_even():
    odd_even = OddEvenResult(
        available=True, depth_odd=0.01, depth_even=0.0095,
        depth_odd_err=0.001, depth_even_err=0.001,
        n_in_transit_odd=10, n_in_transit_even=10,
        significance_sigma=0.5, mismatch=False,
    )
    comparison = _clear_comparison(odd_even=odd_even)

    text = to_text(comparison)
    assert "odd-even depth test: consistent" in text

    payload = json.loads(to_json(comparison))
    assert payload["odd_even"]["available"] is True
    assert payload["odd_even"]["mismatch"] is False


def test_to_text_and_json_include_unavailable_odd_even():
    odd_even = OddEvenResult(available=False, note="insufficient odd/even coverage")
    comparison = _clear_comparison(odd_even=odd_even)

    text = to_text(comparison)
    assert "odd-even depth test: unavailable" in text
    assert "insufficient odd/even coverage" in text

    payload = json.loads(to_json(comparison))
    assert payload["odd_even"] == {"available": False, "note": "insufficient odd/even coverage"}


def test_to_text_flags_mismatch():
    odd_even = OddEvenResult(
        available=True, depth_odd=0.03, depth_even=0.01,
        depth_odd_err=0.001, depth_even_err=0.001,
        n_in_transit_odd=10, n_in_transit_even=10,
        significance_sigma=14.1, mismatch=True,
    )
    comparison = _clear_comparison(odd_even=odd_even)
    text = to_text(comparison)
    assert "odd-even depth test: MISMATCH" in text


def test_to_json_emits_null_not_infinity_for_a_failed_model_fit():
    # A model that never converged (fit_model's total-failure path) reports
    # bic/aic/chi2 as float('inf'); json.dumps' default bare Infinity/NaN
    # tokens aren't valid JSON (RFC 8259) and would break strict parsers
    # consuming this "public output contract".
    failed = _result("starspot", bic=float("inf"), chi2=float("inf"), converged=False)
    ok = _result("planet", bic=60.0, chi2=50.0)
    phase, flux, flux_err = _flat_data()
    comparison = compare([ok, failed], phase, flux, flux_err)

    text = to_json(comparison)
    assert "Infinity" not in text
    assert "NaN" not in text

    payload = json.loads(text)
    starspot = next(m for m in payload["models"] if m["model"] == "starspot")
    assert starspot["bic"] is None
    assert starspot["aic"] is None
    assert starspot["chi2"] is None
    assert starspot["delta_bic"] is None
