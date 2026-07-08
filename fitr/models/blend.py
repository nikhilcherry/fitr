"""Diluted-transit forward model: a transit signal weakened by third light.

Honesty note (see README / compare.py): photometry alone often cannot
distinguish a diluted eclipsing binary from a genuine planet — the real
discriminator is a centroid offset, which fitr never sees. compare.py
flags planet/blend degeneracy explicitly rather than silently picking one.
"""

from __future__ import annotations

import numpy as np

from .base import Model
from .planet import _batman_flux
from ._util import a_rs_from_duration, estimate_depth_and_duration


class BlendModel(Model):
    name = "blend"
    param_names = ["rp_rs", "a_rs", "inc", "t0_shift", "dilution"]

    def bounds(self, lc) -> list[tuple[float, float]]:
        return [
            (1e-4, 0.6),   # rp_rs (allowed deeper than planet.py: diluted EB eclipse)
            (1.5, 200.0),  # a_rs
            (60.0, 90.0),  # inc (deg)
            (-0.05, 0.05), # t0_shift (phase)
            (0.05, 1.0),   # dilution
        ]

    def initial_guess(
        self, phase: np.ndarray, flux: np.ndarray, period: float
    ) -> np.ndarray:
        depth, duration, t0_est = estimate_depth_and_duration(phase, flux)
        # Assume the observed depth is already diluted; guess a deeper
        # intrinsic eclipse with moderate dilution as a starting point.
        rp_rs = float(np.clip(np.sqrt(depth) * 1.5, 1e-3, 0.55))
        a_rs = a_rs_from_duration(duration, inc_deg=89.0)
        return np.array(
            [rp_rs, a_rs, 89.0, np.clip(t0_est, -0.05, 0.05), 0.5]
        )

    def evaluate(
        self, phase: np.ndarray, params: np.ndarray, period: float
    ) -> np.ndarray:
        rp_rs, a_rs, inc, t0_shift, dilution = params
        transit_flux = _batman_flux(phase, rp_rs, a_rs, inc, t0_shift)
        transit_dip = 1.0 - transit_flux
        return 1.0 - dilution * transit_dip
