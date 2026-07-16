"""Model comparison: BIC ranking, ambiguity detection, null-signal gating."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fit import FitResult
from .vetting import OddEvenResult

AMBIGUOUS_DELTA_BIC = 2.0
NO_SIGNAL_DELTA_BIC = 10.0


@dataclass
class Comparison:
    results: list[FitResult]
    baseline_chi2: float
    baseline_bic: float
    delta_bic: dict[str, float]
    winner: str | None
    verdict: str  # "clear" | "ambiguous" | "no_significant_signal"
    tied_models: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    odd_even: OddEvenResult | None = None


def _flat_baseline_chi2(flux: np.ndarray, flux_err: np.ndarray) -> float:
    weights = 1.0 / np.square(flux_err)
    c = np.sum(flux * weights) / np.sum(weights)
    return float(np.sum(np.square((flux - c) / flux_err)))


def compare(
    results: list[FitResult],
    phase: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    odd_even: OddEvenResult | None = None,
) -> Comparison:
    """Rank fitted models by BIC and decide a verdict.

    - "no_significant_signal" if no model beats a 1-parameter flat baseline
      by ΔBIC > 10 (checked first: an insignificant fit should never be
      reported as merely "ambiguous between models").
    - "ambiguous" if the best two (or more) models are within ΔBIC < 2.
      planet/blend ties get an explicit centroid-vetting caveat.
    - "clear" otherwise.

    `odd_even`, if given, contributes its own note whenever it flags a
    mismatch — regardless of verdict, since a period-doubled eclipsing
    binary can just as easily win as a "clear" planet as show up ambiguous.
    """
    n_points = len(flux)
    baseline_chi2 = _flat_baseline_chi2(flux, flux_err)
    baseline_bic = baseline_chi2 + 1.0 * np.log(n_points)

    odd_even_notes = [odd_even.note] if odd_even is not None and odd_even.mismatch else []

    sorted_results = sorted(results, key=lambda r: r.bic)
    finite_results = [r for r in sorted_results if np.isfinite(r.bic)]

    if not finite_results:
        return Comparison(
            results=sorted_results,
            baseline_chi2=baseline_chi2,
            baseline_bic=baseline_bic,
            delta_bic={r.model_name: float("inf") for r in results},
            winner=None,
            verdict="no_significant_signal",
            tied_models=[],
            notes=["all model fits failed to converge", *odd_even_notes],
            odd_even=odd_even,
        )

    best = finite_results[0]
    delta_bic = {
        r.model_name: (r.bic - best.bic if np.isfinite(r.bic) else float("inf"))
        for r in results
    }

    if baseline_bic - best.bic <= NO_SIGNAL_DELTA_BIC:
        return Comparison(
            results=sorted_results,
            baseline_chi2=baseline_chi2,
            baseline_bic=baseline_bic,
            delta_bic=delta_bic,
            winner=None,
            verdict="no_significant_signal",
            tied_models=[],
            notes=[
                f"no model improves on a flat baseline by more than "
                f"ΔBIC > {NO_SIGNAL_DELTA_BIC:.0f}",
                *odd_even_notes,
            ],
            odd_even=odd_even,
        )

    tied_models = [
        r.model_name for r in finite_results if delta_bic[r.model_name] < AMBIGUOUS_DELTA_BIC
    ]

    if len(tied_models) >= 2:
        notes = []
        if "planet" in tied_models and "blend" in tied_models:
            notes.append(
                "planet/blend degenerate: needs centroid vetting "
                "(photometry alone cannot distinguish a diluted eclipsing "
                "binary from a genuine planet)"
            )
        notes.extend(odd_even_notes)
        return Comparison(
            results=sorted_results,
            baseline_chi2=baseline_chi2,
            baseline_bic=baseline_bic,
            delta_bic=delta_bic,
            winner=None,
            verdict="ambiguous",
            tied_models=tied_models,
            notes=notes,
            odd_even=odd_even,
        )

    return Comparison(
        results=sorted_results,
        baseline_chi2=baseline_chi2,
        baseline_bic=baseline_bic,
        delta_bic=delta_bic,
        winner=best.model_name,
        verdict="clear",
        tied_models=[best.model_name],
        notes=odd_even_notes,
        odd_even=odd_even,
    )
