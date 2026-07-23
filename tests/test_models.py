from __future__ import annotations

import numpy as np
import pytest

from fitr.io import LightCurve
from fitr.models import ALL_MODELS


@pytest.fixture(params=list(ALL_MODELS.keys()))
def model(request):
    return ALL_MODELS[request.param]


def _dummy_lc(n=200):
    phase = np.linspace(-0.5, 0.5, n, endpoint=False)
    flux = np.ones(n)
    flux_err = np.full(n, 0.001)
    return phase, flux, flux_err, LightCurve(time=phase, flux=flux, flux_err=flux_err, meta={})


def test_bounds_shape_matches_params(model):
    _, _, _, lc = _dummy_lc()
    bounds = model.bounds(lc)
    assert len(bounds) == len(model.param_names)
    for lo, hi in bounds:
        assert lo < hi


def test_initial_guess_within_bounds(model):
    phase, flux, flux_err, lc = _dummy_lc()
    bounds = model.bounds(lc)
    guess = model.initial_guess(phase, flux, period=3.0)
    assert len(guess) == len(model.param_names)
    for value, (lo, hi) in zip(guess, bounds):
        assert lo - 1e-9 <= value <= hi + 1e-9


def test_evaluate_returns_finite_array(model):
    phase, flux, flux_err, lc = _dummy_lc()
    guess = model.initial_guess(phase, flux, period=3.0)
    out = model.evaluate(phase, guess, period=3.0)
    assert out.shape == phase.shape
    assert np.all(np.isfinite(out))


def test_registry_exposes_all_four():
    assert set(ALL_MODELS.keys()) == {"planet", "eb", "blend", "starspot"}


def test_eb_secondary_never_deeper_than_primary():
    from fitr.models.eb import EBModel

    model = EBModel()
    phase = np.linspace(-0.5, 0.5, 500, endpoint=False)
    # d2 (0.5) requested deeper than d1 (0.1); model must clip d2 <= d1.
    params = np.array([0.1, 0.5, 0.05, 0.05, 0.0, 0.0])
    flux = model.evaluate(phase, params, period=3.0)
    primary_depth = 1.0 - flux[np.argmin(np.abs(phase))]
    secondary_depth = 1.0 - flux[np.argmin(np.abs(phase - 0.5))]
    assert secondary_depth <= primary_depth + 1e-9


def test_eb_ellipsoidal_variation_dips_at_conjunction_peaks_at_quadrature():
    """Real tidally-distorted binaries are faintest at the conjunctions
    (phase 0, 0.5 -- smallest projected stellar area) and brightest at
    quadrature (phase 0.25, 0.75 -- largest projected area); see e.g. the
    OGLE ellipsoidal-variable atlas. With the eclipses themselves switched
    off (d1=d2=0), only the ellipsoidal term should remain."""
    from fitr.models.eb import EBModel

    model = EBModel()
    phase = np.array([0.0, 0.25, 0.5, 0.75])
    params = np.array([1e-5, 0.0, 0.05, 0.05, 0.03, 0.0])
    flux = model.evaluate(phase, params, period=3.0)

    assert flux[0] < 1.0  # conjunction: dip
    assert flux[2] < 1.0  # conjunction: dip
    assert flux[1] > 1.0  # quadrature: brighten
    assert flux[3] > 1.0  # quadrature: brighten
    assert flux[1] > flux[0]
    assert flux[3] > flux[2]
