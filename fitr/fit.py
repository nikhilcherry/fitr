"""Per-model derivative-free optimization (batman is not differentiable)."""

from __future__ import annotations

import time as _time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .io import LightCurve
from .models.base import Model

N_RANDOM_RESTARTS = 4
SEED = 42


@dataclass
class FitResult:
    model_name: str
    params: dict[str, float]
    chi2: float
    n_points: int
    n_params: int
    bic: float
    aic: float
    converged: bool
    runtime_s: float


def _residuals(theta, model, phase, flux, flux_err, period):
    # Exceptions (e.g. a genuinely broken model) are intentionally NOT
    # caught here: they propagate up to the per-start try/except in
    # fit_model, which discards that start rather than reporting a
    # bogus "converged" fit on a flat/fake residual surface. Only
    # non-finite numerical output (e.g. batman instability near a
    # bound) is defended against, so the optimizer can steer away from
    # that region on subsequent iterations.
    model_flux = model.evaluate(phase, theta, period)
    if not np.all(np.isfinite(model_flux)):
        return np.full(len(flux), 1e6)
    return (model_flux - flux) / flux_err


def fit_model(
    model: Model,
    phase: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
) -> FitResult:
    """Fit one model to phase-folded data via scipy least_squares, multi-start."""
    start_time = _time.perf_counter()
    n_points = len(flux)
    n_params = len(model.param_names)

    lc_shim = LightCurve(time=phase, flux=flux, flux_err=flux_err, meta={})
    bounds = model.bounds(lc_shim)
    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)

    heuristic_guess = np.clip(
        model.initial_guess(phase, flux, period), lower, upper
    )

    rng = np.random.default_rng(SEED)
    starts = [heuristic_guess]
    for _ in range(N_RANDOM_RESTARTS):
        starts.append(rng.uniform(lower, upper))

    best_chi2 = np.inf
    best_x = None
    best_converged = False

    for x0 in starts:
        try:
            result = least_squares(
                _residuals,
                x0,
                args=(model, phase, flux, flux_err, period),
                method="trf",
                bounds=(lower, upper),
            )
        except Exception:
            continue

        chi2 = float(np.sum(result.fun**2))
        if not np.isfinite(chi2):
            continue
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_x = result.x
            best_converged = bool(result.success)

    runtime_s = _time.perf_counter() - start_time

    if best_x is None:
        return FitResult(
            model_name=model.name,
            params={name: float("nan") for name in model.param_names},
            chi2=float("inf"),
            n_points=n_points,
            n_params=n_params,
            bic=float("inf"),
            aic=float("inf"),
            converged=False,
            runtime_s=runtime_s,
        )

    bic = best_chi2 + n_params * np.log(n_points)
    aic = best_chi2 + 2 * n_params

    return FitResult(
        model_name=model.name,
        params=model.params_to_dict(best_x),
        chi2=best_chi2,
        n_points=n_points,
        n_params=n_params,
        bic=float(bic),
        aic=float(aic),
        converged=best_converged,
        runtime_s=runtime_s,
    )


def fit_all(lc: LightCurve, period: float, epoch: float) -> list[FitResult]:
    """Fold `lc` and fit all registered models to it."""
    from .fold import fold, maybe_autobin
    from .models import ALL_MODELS

    if period <= 0:
        # fold() divides by period; period<=0 silently turns every phase
        # into NaN/inf, and the flat-phase degenerate case that follows
        # can spuriously produce a *lower* chi2 than a real fit would --
        # a confidently wrong "clear winner" instead of a usage error.
        # This validation lives here (not just in cli.py) because fit_all
        # is the public library entry point too.
        raise ValueError(f"period must be positive, got {period}")

    phase = fold(lc.time, period, epoch)
    phase, flux, flux_err = maybe_autobin(phase, lc.flux, lc.flux_err)

    return [
        fit_model(model, phase, flux, flux_err, period)
        for model in ALL_MODELS.values()
    ]
