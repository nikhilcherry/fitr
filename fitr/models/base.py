"""Abstract base class for forward models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    """A physical forward model of a phase-folded light curve.

    Subclasses model flux purely as a function of phase; the real orbital
    `period` is passed through for interface uniformity but transit shape
    in phase-space is independent of it (duration/period depends only on
    a_rs, inc, rp_rs — the real period cancels out of the phase-domain
    Kepler solution), so most models ignore it.
    """

    name: str
    param_names: list[str]

    @abstractmethod
    def bounds(self, lc) -> list[tuple[float, float]]:
        """Return (lower, upper) bounds for each parameter, in param_names order."""

    @abstractmethod
    def initial_guess(
        self, phase: np.ndarray, flux: np.ndarray, period: float
    ) -> np.ndarray:
        """Return an initial parameter vector, in param_names order."""

    @abstractmethod
    def evaluate(
        self, phase: np.ndarray, params: np.ndarray, period: float
    ) -> np.ndarray:
        """Return model flux at the given phases for the given parameter vector."""

    def params_to_dict(self, params: np.ndarray) -> dict[str, float]:
        return {name: float(value) for name, value in zip(self.param_names, params)}
