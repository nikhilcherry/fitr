"""Human-readable and JSON formatting of a Comparison. JSON is the public
output contract for fitr — see README for the documented schema."""

from __future__ import annotations

import json
import math

from .compare import Comparison


def _round_sig(x: float, sig: int = 6) -> float:
    if x is None or not math.isfinite(x):
        return x
    if x == 0:
        return 0.0
    digits = sig - int(math.floor(math.log10(abs(x)))) - 1
    return round(x, digits)


def _round_dict(d: dict) -> dict:
    return {k: _round_sig(v) for k, v in d.items()}


def to_json(comparison: Comparison) -> str:
    payload = {
        "verdict": comparison.verdict,
        "winner": comparison.winner,
        "tied_models": comparison.tied_models,
        "baseline_chi2": _round_sig(comparison.baseline_chi2),
        "baseline_bic": _round_sig(comparison.baseline_bic),
        "models": [
            {
                "model": r.model_name,
                "converged": r.converged,
                "chi2": _round_sig(r.chi2),
                "bic": _round_sig(r.bic),
                "aic": _round_sig(r.aic),
                "delta_bic": _round_sig(comparison.delta_bic.get(r.model_name, float("inf"))),
                "n_points": r.n_points,
                "n_params": r.n_params,
                "runtime_s": _round_sig(r.runtime_s),
                "params": _round_dict(r.params),
            }
            for r in comparison.results
        ],
        "notes": comparison.notes,
    }
    return json.dumps(payload, indent=2)


def to_text(comparison: Comparison) -> str:
    header = f"{'model':<10} {'chi2':>12} {'BIC':>12} {'dBIC':>10} {'converged':>10}"
    lines = [header, "-" * len(header)]
    for r in comparison.results:
        dbic = comparison.delta_bic.get(r.model_name, float("inf"))
        lines.append(
            f"{r.model_name:<10} {r.chi2:>12.3f} {r.bic:>12.3f} {dbic:>10.3f} "
            f"{str(r.converged):>10}"
        )

    lines.append("")
    lines.append(f"baseline (flat, 1 param): chi2={comparison.baseline_chi2:.3f} "
                  f"bic={comparison.baseline_bic:.3f}")
    lines.append("")

    if comparison.verdict == "clear":
        lines.append(f"verdict: clear winner = {comparison.winner}")
    elif comparison.verdict == "ambiguous":
        lines.append(f"verdict: ambiguous, tied models = {', '.join(comparison.tied_models)}")
    else:
        lines.append("verdict: no_significant_signal")

    for note in comparison.notes:
        lines.append(f"note: {note}")

    return "\n".join(lines)
