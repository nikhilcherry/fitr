"""Transiting-planet forward model, via batman.

Known v1 simplifications (see README): circular orbit (ecc=0 fixed),
fixed quadratic limb darkening u=[0.4, 0.25] (typical TESS bandpass,
not fit — batman is not differentiable so LD fitting would be slow and
poorly constrained by a single-band light curve anyway).
"""

from __future__ import annotations

import batman
import numpy as np

from .base import Model
from ._util import a_rs_from_duration, estimate_depth_and_duration

LD_COEFFS = [0.4, 0.25]


def _batman_flux(
    phase: np.ndarray, rp_rs: float, a_rs: float, inc: float, t0_shift: float
) -> np.ndarray:
    params = batman.TransitParams()
    params.t0 = t0_shift
    params.per = 1.0
    params.rp = rp_rs
    params.a = a_rs
    params.inc = inc
    params.ecc = 0.0
    params.w = 90.0
    params.u = LD_COEFFS
    params.limb_dark = "quadratic"

    model = batman.TransitModel(params, phase)
    return model.light_curve(params)


class PlanetModel(Model):
    name = "planet"
    param_names = ["rp_rs", "a_rs", "inc", "t0_shift"]

    def bounds(self, lc) -> list[tuple[float, float]]:
        return [
            (1e-4, 0.3),   # rp_rs
            (1.5, 200.0),  # a_rs
            (60.0, 90.0),  # inc (deg)
            (-0.05, 0.05), # t0_shift (phase)
        ]

    def initial_guess(
        self, phase: np.ndarray, flux: np.ndarray, period: float
    ) -> np.ndarray:
        depth, duration, t0_est = estimate_depth_and_duration(phase, flux)
        rp_rs = float(np.clip(np.sqrt(depth), 1e-3, 0.29))
        a_rs = a_rs_from_duration(duration, inc_deg=89.0)
        return np.array([rp_rs, a_rs, 89.0, np.clip(t0_est, -0.05, 0.05)])

    def evaluate(
        self, phase: np.ndarray, params: np.ndarray, period: float
    ) -> np.ndarray:
        rp_rs, a_rs, inc, t0_shift = params
        return _batman_flux(phase, rp_rs, a_rs, inc, t0_shift)
