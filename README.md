# fitr

`fitr` fits four competing physical forward models — **planet**,
**eclipsing binary (eb)**, **blend**, and **starspot** — to a phase-folded
light curve, and reports which model explains the data best. The
best-fitting model *is* the classification: this is the
analysis-by-synthesis core of the Arvyo BAH2026 pipeline ("Explain, Not
Just Classify"), the sibling tool to
[`foldr`](https://github.com/nikhilcherry/foldr) (period search) and
[`batchr`](https://github.com/nikhilcherry/batchr) (bulk processing).

## Install

```bash
pip install git+https://github.com/nikhilcherry/fitr
```

> **Python 3.12+ note:** `fitr` depends on
> [`batman-package`](https://pypi.org/project/batman-package/), which
> still imports the standard-library `distutils` module removed in
> Python 3.12. If you see `ModuleNotFoundError: No module named
> 'distutils'` on import, run `pip install setuptools` in the same
> environment — most environments already have it, but a bare `python -m
> venv` on 3.12+ does not install it by default.

## Quickstart

```bash
fitr fit lightcurve.npz --period 3.14 --epoch 0.5
```

```
model              chi2          BIC       dBIC  converged
----------------------------------------------------------
planet          408.944      432.910      0.000       True
blend           408.944      440.511      7.601       True
eb              612.331      646.938    214.028       True
starspot       3820.442     3838.311   3405.401       True

baseline (flat, 1 param): chi2=3948.500 bic=3954.490

verdict: clear winner = planet
```

If the input `.npz` carries `period_days` / `epoch_btjd` in its metadata,
`--period` / `--epoch` may be omitted and fitr will use them (announced on
stderr). With neither source available, fitr exits 2 with `no period
available — run foldr first to find one.`

Machine-readable output:

```bash
fitr fit lightcurve.npz --period 3.14 --epoch 0.5 --json
```

## Workflow

```bash
foldr lightcurve.fits                       # find a period
fitr fit lightcurve.npz --period 3.14        # classify by model fit
batchr run manifest.csv --tool fitr          # bulk-run fitr over many curves
```

## The four models

| model | params | what it captures |
|---|---|---|
| `planet` | `rp_rs, a_rs, inc, t0_shift` | single transiting planet (via [batman](https://github.com/lkreidberg/batman)) |
| `eb` | `d1, d2, dur1, dur2, amp_ellip, t0_shift` | primary + secondary eclipse (analytic trapezoids) + ellipsoidal variation |
| `blend` | `rp_rs, a_rs, inc, t0_shift, dilution` | a transit diluted by third light |
| `starspot` | `a1, phi1, a2, phi2` | rotational modulation, no eclipse at all (two-harmonic sinusoid) |

**Stated simplifications (v1):**
- Limb darkening is fixed quadratic `u=[0.4, 0.25]` (typical TESS
  bandpass) and never fit — batman is not differentiable, so
  derivative-free LD fitting on a single band would be slow and poorly
  constrained anyway.
- Orbits are circular (`ecc=0` fixed) for `planet` and `blend`.
- `eb` has no odd-even depth test; that needs unfolded data at 2× the
  period. Future work.
- **`blend` vs `planet` is often genuinely degenerate from photometry
  alone** — the real discriminator is a centroid offset, which fitr never
  sees. When their BIC scores are within `ΔBIC < 2`, fitr reports
  `"ambiguous"` with an explicit centroid-vetting note rather than
  silently picking a winner.
- All fitting is derivative-free (`scipy.optimize.least_squares`,
  numeric Jacobian) with 5 seeded starts (1 heuristic + 4 random,
  `numpy.random.default_rng(42)`); results are exactly reproducible
  run-to-run.

## JSON output schema

```jsonc
{
  "verdict": "clear",              // "clear" | "ambiguous" | "no_significant_signal"
  "winner": "planet",              // model name, or null if not "clear"
  "tied_models": ["planet"],       // models within ΔBIC < 2 of the best (ambiguous case)
  "baseline_chi2": 3948.5,         // chi2 of a 1-parameter flat-flux fit
  "baseline_bic": 3954.49,
  "models": [
    {
      "model": "planet",
      "converged": true,
      "chi2": 408.944,
      "bic": 432.91,
      "aic": 416.944,
      "delta_bic": 0.0,           // bic - best_bic
      "n_points": 400,
      "n_params": 4,
      "runtime_s": 0.0276,
      "params": { "rp_rs": 0.12, "a_rs": 10.01, "inc": 88.5, "t0_shift": 0.0003 }
    }
    // ... eb, blend, starspot, sorted by bic ascending
  ],
  "notes": []                      // e.g. planet/blend centroid-vetting caveat
}
```

All floats are rounded to 6 significant figures. Two identical runs
produce byte-identical JSON.

## Exit codes

| code | meaning |
|---|---|
| 0 | clear winner |
| 2 | usage or input error (bad file, no period available) |
| 3 | ambiguous (tied models within ΔBIC < 2) |
| 4 | no_significant_signal (no model beats a flat baseline by ΔBIC > 10) |

`batchr` keys off these codes when driving bulk runs.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

`tests/data/sample_planet_synthetic.npz` is a small (400-point) seeded
synthetic planet-transit injection used as a test fixture — it is not
real telescope data.
