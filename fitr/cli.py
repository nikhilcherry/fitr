"""Command-line interface for fitr."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from .compare import compare
from .fit import fit_all
from .fold import fold as fold_phase
from .fold import maybe_autobin
from .io import FitrInputError, load_lightcurve
from .models import ALL_MODELS
from .report import to_json, to_text
from .vetting import odd_even_test

EXIT_CLEAR = 0
EXIT_USAGE_ERROR = 2
EXIT_AMBIGUOUS = 3
EXIT_NO_SIGNAL = 4

_VERDICT_EXIT_CODES = {
    "clear": EXIT_CLEAR,
    "ambiguous": EXIT_AMBIGUOUS,
    "no_significant_signal": EXIT_NO_SIGNAL,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fitr",
        description="Fit competing physical forward models to a phase-folded "
        "light curve: planet / eclipsing binary / blend / starspot.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser(
        "fit", help="Fit all models to a light curve and report the best."
    )
    fit_parser.add_argument("lightcurve", help="Path to a .npz or .csv light curve file")
    fit_parser.add_argument(
        "--period", type=float, default=None, help="Orbital period (days)"
    )
    fit_parser.add_argument(
        "--epoch", type=float, default=None, help="Transit epoch (same time units as file)"
    )
    fit_parser.add_argument(
        "--json", action="store_true", dest="json_output", help="Print JSON report to stdout"
    )
    fit_parser.add_argument(
        "--out", type=str, default=None, metavar="PATH", help="Also write JSON report to PATH"
    )

    subparsers.add_parser("models", help="List available models and their parameters.")
    subparsers.add_parser("version", help="Print the fitr version.")

    return parser


def _cmd_models() -> int:
    for name, model in ALL_MODELS.items():
        print(f"{name}: {', '.join(model.param_names)}")
    return EXIT_CLEAR


def _cmd_version() -> int:
    print(f"fitr {__version__}")
    return EXIT_CLEAR


def _cmd_fit(args: argparse.Namespace) -> int:
    path = os.path.abspath(os.path.join(os.getcwd(), args.lightcurve))

    try:
        lc = load_lightcurve(path)
    except FitrInputError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    period = args.period
    epoch = args.epoch

    if period is None and "period_days" in lc.meta:
        period = lc.meta["period_days"]
        print(f"Note: using period_days={period} from file metadata", file=sys.stderr)
    if epoch is None and "epoch_btjd" in lc.meta:
        epoch = lc.meta["epoch_btjd"]
        print(f"Note: using epoch_btjd={epoch} from file metadata", file=sys.stderr)

    if period is None:
        print("no period available — run foldr first to find one.", file=sys.stderr)
        return EXIT_USAGE_ERROR
    if epoch is None:
        epoch = float(lc.time[0])
        print(f"Note: no epoch given, defaulting to first timestamp {epoch}", file=sys.stderr)

    results = fit_all(lc, period, epoch)

    phase = fold_phase(lc.time, period, epoch)
    phase, flux, flux_err = maybe_autobin(phase, lc.flux, lc.flux_err)
    odd_even = odd_even_test(lc.time, lc.flux, lc.flux_err, period, epoch)
    comparison = compare(results, phase, flux, flux_err, odd_even=odd_even)

    if args.json_output:
        print(to_json(comparison))
    else:
        print(to_text(comparison))

    if args.out:
        with open(args.out, "w") as f:
            f.write(to_json(comparison))

    return _VERDICT_EXIT_CODES.get(comparison.verdict, EXIT_USAGE_ERROR)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "fit":
        return _cmd_fit(args)
    if args.command == "models":
        return _cmd_models()
    if args.command == "version":
        return _cmd_version()

    parser.print_help()
    return EXIT_USAGE_ERROR


if __name__ == "__main__":
    sys.exit(main())
