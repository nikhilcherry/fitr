"""Eclipsing-binary forward model: two trapezoidal eclipses + ellipsoidal term.

Implemented with analytic trapezoids rather than batman: an EB's eclipses
are not well described by a single-star-transiting-star batman call when
both depths, both durations, and a secondary can vary independently, and
a trapezoid is cheap, robust to non-convergence, and captures the two key
discriminators this model needs (unequal depths, V-shaped/grazing
eclipses via duration-depth freedom). Odd-even depth differences require
unfolded data at 2x the period and are out of scope for v1 (future work).
"""

from __future__ import annotations

import numpy as np

from .base import Model

INGRESS_FRACTION = 0.2  # fixed fraction of duration spent in ingress+egress


def _wrapped_distance(phase: np.ndarray, center: float) -> np.ndarray:
    d = np.abs(phase - center)
    return np.minimum(d, 1.0 - d)


def _trapezoid(phase: np.ndarray, center: float, duration: float, depth: float) -> np.ndarray:
    duration = max(duration, 1e-4)
    dist = _wrapped_distance(phase, center % 1.0)
    half_dur = duration / 2.0
    half_flat = half_dur * (1.0 - INGRESS_FRACTION)
    slope_width = half_dur - half_flat

    dip = np.zeros_like(phase)
    flat_mask = dist <= half_flat
    dip[flat_mask] = depth

    if slope_width > 0:
        slope_mask = (dist > half_flat) & (dist <= half_dur)
        frac = (half_dur - dist[slope_mask]) / slope_width
        dip[slope_mask] = depth * frac

    return dip


class EBModel(Model):
    name = "eb"
    param_names = ["d1", "d2", "dur1", "dur2", "amp_ellip", "t0_shift"]

    def bounds(self, lc) -> list[tuple[float, float]]:
        return [
            (1e-5, 0.9),    # d1
            (0.0, 0.9),     # d2 (clipped to <= d1 at evaluation time)
            (0.005, 0.4),   # dur1
            (0.005, 0.4),   # dur2
            (0.0, 0.05),    # amp_ellip
            (-0.05, 0.05),  # t0_shift
        ]

    def initial_guess(
        self, phase: np.ndarray, flux: np.ndarray, period: float
    ) -> np.ndarray:
        from ._util import estimate_depth_and_duration

        depth1, dur1, t0_est = estimate_depth_and_duration(phase, flux)

        secondary_phase = (phase - 0.5) % 1.0
        secondary_phase = np.where(secondary_phase >= 0.5, secondary_phase - 1.0, secondary_phase)
        depth2, dur2, _ = estimate_depth_and_duration(secondary_phase, flux)
        depth2 = min(depth2, depth1 * 0.9)

        return np.array(
            [
                float(np.clip(depth1, 1e-4, 0.85)),
                float(np.clip(depth2, 0.0, 0.85)),
                float(np.clip(dur1, 0.01, 0.35)),
                float(np.clip(dur2, 0.01, 0.35)),
                0.001,
                float(np.clip(t0_est, -0.05, 0.05)),
            ]
        )

    def evaluate(
        self, phase: np.ndarray, params: np.ndarray, period: float
    ) -> np.ndarray:
        d1, d2, dur1, dur2, amp_ellip, t0_shift = params
        d2 = min(d2, d1)

        flux = np.ones_like(phase)
        flux -= _trapezoid(phase, t0_shift, dur1, d1)
        flux -= _trapezoid(phase, 0.5 + t0_shift, dur2, d2)
        flux += amp_ellip * np.cos(4.0 * np.pi * phase)
        return flux
