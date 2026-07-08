"""Regenerate the README illustration PNGs under assets/.

Not part of the installed package and not run by tests — matplotlib is
intentionally excluded from fitr's runtime dependencies (numpy, scipy,
batman-package only). Run manually after model/fit changes:

    pip install matplotlib
    python scripts/generate_assets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fitr.compare import compare
from fitr.fit import fit_model
from fitr.fold import bin_folded, fold
from fitr.models import ALL_MODELS
from tests.conftest import (
    make_blend_curve,
    make_eb_curve,
    make_planet_curve,
    make_starspot_curve,
)

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
MODEL_COLORS = {
    "planet": "#2166ac",
    "eb": "#b2182b",
    "blend": "#762a83",
    "starspot": "#1b7837",
}


def _fit_and_fold(sc):
    phase = fold(sc.lc.time, sc.period, sc.epoch)
    results = [
        fit_model(model, phase, sc.lc.flux, sc.lc.flux_err, sc.period)
        for model in ALL_MODELS.values()
    ]
    comparison = compare(results, phase, sc.lc.flux, sc.lc.flux_err)
    return phase, results, comparison


def _plot_panel(ax, sc, phase, results, comparison, show_legend=False):
    centers, means, errs = bin_folded(phase, sc.lc.flux, sc.lc.flux_err, n_bins=60)
    ax.errorbar(
        centers, means, yerr=errs, fmt="o", ms=3, color="0.35",
        ecolor="0.75", elinewidth=1, capsize=0, zorder=2, label="data (binned)",
    )

    fine_phase = np.linspace(-0.5, 0.5, 2000, endpoint=False)
    model_by_name = {r.model_name: r for r in results}
    for name, model in ALL_MODELS.items():
        r = model_by_name[name]
        params = np.array([r.params[p] for p in model.param_names])
        fine_flux = model.evaluate(fine_phase, params, sc.period)
        is_winner = name == comparison.winner or name in comparison.tied_models
        ax.plot(
            fine_phase, fine_flux,
            color=MODEL_COLORS[name],
            linewidth=2.5 if is_winner else 1.2,
            alpha=1.0 if is_winner else 0.55,
            linestyle="-" if is_winner else "--",
            zorder=3 if is_winner else 1,
            label=f"{name} fit" + ("  (winner)" if is_winner else ""),
        )

    title = f"true = {sc.true_model}   |   fitr verdict = {comparison.verdict}"
    if comparison.winner:
        title += f" ({comparison.winner})"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("orbital phase")
    ax.set_ylabel("normalized flux")
    ax.set_xlim(-0.5, 0.5)
    if show_legend:
        ax.legend(fontsize=7, loc="lower right", ncol=2, framealpha=0.9)


def make_model_gallery():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    makers = [make_planet_curve, make_eb_curve, make_blend_curve, make_starspot_curve]
    for ax, maker in zip(axes.flat, makers):
        sc = maker()
        phase, results, comparison = _fit_and_fold(sc)
        _plot_panel(ax, sc, phase, results, comparison, show_legend=(maker is make_planet_curve))

    fig.suptitle(
        "fitr: four forward models fit to four synthetic light curves\n"
        "(analysis by synthesis — the best-fitting model is the classification)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = ASSETS_DIR / "model_gallery.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def make_degenerate_example():
    """A genuine planet/blend tie: a deep, grazing eclipse (rp_rs=0.4, an
    EB-like depth outside the planet model's own bounds) heavily diluted
    (dilution=0.3) down to a modest apparent depth. Found by grid search
    over fit_all + compare (see git history) to land within ΔBIC < 2 —
    this is a real fit output, not a fabricated one, chosen only for the
    parameters that produce it."""
    from fitr.io import LightCurve
    import batman

    period, epoch = 3.0, 1.5
    rng = np.random.default_rng(1)
    time = np.sort(rng.uniform(0, 30, 250))

    params = batman.TransitParams()
    params.t0 = epoch
    params.per = period
    params.rp = 0.4
    params.a = 4.0
    params.inc = 84.0
    params.ecc = 0.0
    params.w = 90.0
    params.u = [0.4, 0.25]
    params.limb_dark = "quadratic"
    model = batman.TransitModel(params, time)
    transit_flux = model.light_curve(params)
    dip = 1.0 - transit_flux
    dilution = 0.3
    flux = 1.0 - dilution * dip
    flux = flux + rng.normal(0, 0.0006, size=flux.shape)
    flux_err = np.full(flux.shape, 0.0006)

    from tests.conftest import SyntheticCurve

    sc = SyntheticCurve(
        lc=LightCurve(time=time, flux=flux, flux_err=flux_err, meta={}),
        period=period, epoch=epoch, true_model="blend (deep eclipse, diluted 0.3x)",
        true_params={"dilution": dilution, "rp_rs": 0.4},
    )

    phase, results, comparison = _fit_and_fold(sc)

    import textwrap

    fig, ax = plt.subplots(figsize=(8, 5.5))
    _plot_panel(ax, sc, phase, results, comparison, show_legend=True)
    ax.set_title(
        f"planet vs. blend, ΔBIC={comparison.delta_bic.get('blend', float('nan')):.2f}, "
        f"verdict = {comparison.verdict}",
        fontsize=10,
    )
    caption = "\n".join(comparison.notes) or ", ".join(comparison.tied_models)
    fig.text(
        0.5, 0.005, textwrap.fill(caption, width=90),
        ha="center", va="bottom", fontsize=8, style="italic", wrap=True,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = ASSETS_DIR / "degenerate_blend_planet.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    ASSETS_DIR.mkdir(exist_ok=True)
    make_model_gallery()
    make_degenerate_example()
