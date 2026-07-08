"""Rotational-modulation forward model: no eclipse, just a two-harmonic sinusoid."""

from __future__ import annotations

import numpy as np

from .base import Model


class StarspotModel(Model):
    name = "starspot"
    param_names = ["a1", "phi1", "a2", "phi2"]

    def bounds(self, lc) -> list[tuple[float, float]]:
        flux_scatter = float(np.std(lc.flux)) if lc is not None else 0.05
        amp_max = max(0.5, 10.0 * flux_scatter)
        return [
            (0.0, amp_max),        # a1
            (-np.pi, np.pi),       # phi1
            (0.0, amp_max),        # a2
            (-np.pi, np.pi),       # phi2
        ]

    def initial_guess(
        self, phase: np.ndarray, flux: np.ndarray, period: float
    ) -> np.ndarray:
        amp = float(np.clip((np.max(flux) - np.min(flux)) / 2.0, 1e-4, 0.5))
        return np.array([amp, 0.0, amp * 0.3, 0.0])

    def evaluate(
        self, phase: np.ndarray, params: np.ndarray, period: float
    ) -> np.ndarray:
        a1, phi1, a2, phi2 = params
        return (
            1.0
            + a1 * np.sin(2.0 * np.pi * phase + phi1)
            + a2 * np.sin(4.0 * np.pi * phase + phi2)
        )
