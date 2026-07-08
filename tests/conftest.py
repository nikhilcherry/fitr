"""Seeded synthetic light curve generators, one per model class + null."""

from __future__ import annotations

from dataclasses import dataclass

import batman
import numpy as np
import pytest

from fitr.io import LightCurve

PERIOD = 3.0
EPOCH = 1.5
N_POINTS = 2000
TIME_SPAN = 30.0
NOISE_SIGMA = 0.0005


@dataclass
class SyntheticCurve:
    lc: LightCurve
    period: float
    epoch: float
    true_model: str
    true_params: dict


def _time_grid(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(0, TIME_SPAN, N_POINTS))


def _noisy(flux: np.ndarray, seed: int, sigma: float = NOISE_SIGMA) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 1)
    noisy_flux = flux + rng.normal(0, sigma, size=flux.shape)
    flux_err = np.full(flux.shape, sigma)
    return noisy_flux, flux_err


def make_planet_curve(seed: int = 100) -> SyntheticCurve:
    time = _time_grid(seed)
    true_params = {"rp_rs": 0.1, "a_rs": 12.0, "inc": 89.0}

    params = batman.TransitParams()
    params.t0 = EPOCH
    params.per = PERIOD
    params.rp = true_params["rp_rs"]
    params.a = true_params["a_rs"]
    params.inc = true_params["inc"]
    params.ecc = 0.0
    params.w = 90.0
    params.u = [0.4, 0.25]
    params.limb_dark = "quadratic"
    model = batman.TransitModel(params, time)
    flux = model.light_curve(params)

    flux, flux_err = _noisy(flux, seed)
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err, meta={})
    return SyntheticCurve(lc, PERIOD, EPOCH, "planet", true_params)


def make_eb_curve(seed: int = 200) -> SyntheticCurve:
    time = _time_grid(seed)
    phase = ((time - EPOCH) / PERIOD) % 1.0
    phase = np.where(phase >= 0.5, phase - 1.0, phase)

    d1, d2, dur1, dur2 = 0.08, 0.03, 0.04, 0.04
    true_params = {"d1": d1, "d2": d2, "dur1": dur1, "dur2": dur2}

    def trapezoid(center, duration, depth):
        dist = np.abs(phase - center)
        dist = np.minimum(dist, 1.0 - dist)
        half_dur = duration / 2.0
        half_flat = half_dur * 0.8
        slope_width = half_dur - half_flat
        dip = np.zeros_like(phase)
        dip[dist <= half_flat] = depth
        slope_mask = (dist > half_flat) & (dist <= half_dur)
        dip[slope_mask] = depth * (half_dur - dist[slope_mask]) / slope_width
        return dip

    flux = np.ones_like(phase)
    flux -= trapezoid(0.0, dur1, d1)
    flux -= trapezoid(0.5, dur2, d2)

    flux, flux_err = _noisy(flux, seed)
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err, meta={})
    return SyntheticCurve(lc, PERIOD, EPOCH, "eb", true_params)


def make_blend_curve(seed: int = 300) -> SyntheticCurve:
    time = _time_grid(seed)
    true_params = {"rp_rs": 0.35, "a_rs": 8.0, "inc": 88.0, "dilution": 0.2}

    params = batman.TransitParams()
    params.t0 = EPOCH
    params.per = PERIOD
    params.rp = true_params["rp_rs"]
    params.a = true_params["a_rs"]
    params.inc = true_params["inc"]
    params.ecc = 0.0
    params.w = 90.0
    params.u = [0.4, 0.25]
    params.limb_dark = "quadratic"
    model = batman.TransitModel(params, time)
    transit_flux = model.light_curve(params)
    dip = 1.0 - transit_flux
    flux = 1.0 - true_params["dilution"] * dip

    flux, flux_err = _noisy(flux, seed)
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err, meta={})
    return SyntheticCurve(lc, PERIOD, EPOCH, "blend", true_params)


def make_starspot_curve(seed: int = 400) -> SyntheticCurve:
    time = _time_grid(seed)
    phase = ((time - EPOCH) / PERIOD) % 1.0
    phase = np.where(phase >= 0.5, phase - 1.0, phase)

    a1, phi1, a2, phi2 = 0.02, 0.3, 0.006, -0.5
    true_params = {"a1": a1, "phi1": phi1, "a2": a2, "phi2": phi2}
    flux = 1.0 + a1 * np.sin(2 * np.pi * phase + phi1) + a2 * np.sin(4 * np.pi * phase + phi2)

    flux, flux_err = _noisy(flux, seed)
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err, meta={})
    return SyntheticCurve(lc, PERIOD, EPOCH, "starspot", true_params)


def make_null_curve(seed: int = 500) -> SyntheticCurve:
    time = _time_grid(seed)
    flux = np.ones_like(time)
    flux, flux_err = _noisy(flux, seed)
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err, meta={})
    return SyntheticCurve(lc, PERIOD, EPOCH, "null", {})


@pytest.fixture
def planet_curve() -> SyntheticCurve:
    return make_planet_curve()


@pytest.fixture
def eb_curve() -> SyntheticCurve:
    return make_eb_curve()


@pytest.fixture
def blend_curve() -> SyntheticCurve:
    return make_blend_curve()


@pytest.fixture
def starspot_curve() -> SyntheticCurve:
    return make_starspot_curve()


@pytest.fixture
def null_curve() -> SyntheticCurve:
    return make_null_curve()
