"""Odd-even transit depth test: a classic eclipsing-binary vetting check.

An EB whose primary and secondary eclipses have similar depth can fold
convincingly at *half* its true period, masquerading as a single-depth
"planet" transit. Splitting transits by odd/even cycle number and
comparing their depths (computed independently, at full time resolution,
not from the phase-folded/binned data the model fits use) catches this:
a period-halved EB shows a significant depth difference between its odd
and even transits, while a genuine planet does not.

This needs unfolded `time` at the period/epoch used for fitting, which is
exactly what README.md flagged as "future work" for the `eb` model — but
the check is useful for any winning model, not just `eb`, since the
classic failure mode it catches (planet-classified half-period EB) shows
up as a false "planet" verdict, not a false "eb" one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fold import fold

MISMATCH_SIGMA_THRESHOLD = 3.0
MIN_POINTS_PER_PARITY = 5


@dataclass
class OddEvenResult:
    available: bool
    depth_odd: float | None = None
    depth_even: float | None = None
    depth_odd_err: float | None = None
    depth_even_err: float | None = None
    n_in_transit_odd: int = 0
    n_in_transit_even: int = 0
    significance_sigma: float | None = None
    mismatch: bool = False
    note: str | None = None


def _weighted_mean_and_err(values: np.ndarray, errs: np.ndarray) -> tuple[float, float]:
    weights = 1.0 / np.square(errs)
    w_sum = np.sum(weights)
    mean = float(np.sum(values * weights) / w_sum)
    err = float(np.sqrt(1.0 / w_sum))
    return mean, err


def _parity_depth(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    epoch: float,
    half_duration_phase: float,
) -> tuple[float, float, int]:
    """Weighted-mean-baseline minus weighted-mean-in-transit depth for one
    odd/even subset, at full time resolution (no binning)."""
    if len(time) == 0:
        return float("nan"), float("nan"), 0

    phase = fold(time, period, epoch)
    in_mask = np.abs(phase) <= half_duration_phase
    out_mask = ~in_mask
    n_in = int(np.count_nonzero(in_mask))

    if n_in < MIN_POINTS_PER_PARITY or np.count_nonzero(out_mask) < MIN_POINTS_PER_PARITY:
        return float("nan"), float("nan"), n_in

    baseline, baseline_err = _weighted_mean_and_err(flux[out_mask], flux_err[out_mask])
    in_transit, in_transit_err = _weighted_mean_and_err(flux[in_mask], flux_err[in_mask])
    depth = baseline - in_transit
    depth_err = float(np.hypot(baseline_err, in_transit_err))
    return depth, depth_err, n_in


def odd_even_test(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    epoch: float,
) -> OddEvenResult:
    """Compare eclipse/transit depth between odd- and even-numbered cycles.

    Cycle number is `round((time - epoch) / period)`; the in-transit window
    half-width is estimated once from the full (unsplit) folded curve via
    the same coarse-binning heuristic the model fitters use for their
    initial guesses, so this needs no fitted model result as input.
    """
    from .models._util import estimate_depth_and_duration

    if period <= 0 or len(time) == 0:
        return OddEvenResult(available=False, note="no period or no data to test")

    full_phase = fold(time, period, epoch)
    _, duration_phase, _ = estimate_depth_and_duration(full_phase, flux)
    half_duration_phase = duration_phase / 2.0

    cycle = np.round((time - epoch) / period)
    odd_mask = np.mod(cycle, 2) != 0
    even_mask = ~odd_mask

    depth_odd, err_odd, n_odd = _parity_depth(
        time[odd_mask], flux[odd_mask], flux_err[odd_mask], period, epoch, half_duration_phase
    )
    depth_even, err_even, n_even = _parity_depth(
        time[even_mask], flux[even_mask], flux_err[even_mask], period, epoch, half_duration_phase
    )

    if not (np.isfinite(depth_odd) and np.isfinite(depth_even)):
        return OddEvenResult(
            available=False,
            n_in_transit_odd=n_odd,
            n_in_transit_even=n_even,
            note=(
                f"insufficient odd/even coverage for the odd-even depth test "
                f"(need >= {MIN_POINTS_PER_PARITY} in-transit points on each side; "
                f"got odd={n_odd}, even={n_even})"
            ),
        )

    sigma = float(np.hypot(err_odd, err_even))
    significance = abs(depth_odd - depth_even) / sigma if sigma > 0 else 0.0
    mismatch = significance > MISMATCH_SIGMA_THRESHOLD

    note = None
    if mismatch:
        note = (
            f"odd-even depth mismatch ({significance:.1f}σ): odd and even transits "
            "have significantly different depths — possible period-doubled "
            "eclipsing binary rather than a genuine transiting planet"
        )

    return OddEvenResult(
        available=True,
        depth_odd=depth_odd,
        depth_even=depth_even,
        depth_odd_err=err_odd,
        depth_even_err=err_even,
        n_in_transit_odd=n_odd,
        n_in_transit_even=n_even,
        significance_sigma=significance,
        mismatch=mismatch,
        note=note,
    )
