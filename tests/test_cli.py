from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from tests.conftest import make_null_curve, make_planet_curve


def _write_npz(tmp_path, sc, name="lc.npz"):
    path = tmp_path / name
    np.savez(
        path,
        time=sc.lc.time,
        flux=sc.lc.flux,
        flux_err=sc.lc.flux_err,
    )
    return path


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "fitr.cli", *args],
        capture_output=True,
        text=True,
    )


def test_cli_models_command():
    result = _run_cli("models")
    assert result.returncode == 0
    assert "planet" in result.stdout
    assert "starspot" in result.stdout


def test_cli_version_command():
    result = _run_cli("version")
    assert result.returncode == 0
    assert "fitr" in result.stdout


def test_cli_fit_clear_winner_exit_0(tmp_path):
    sc = make_planet_curve()
    path = _write_npz(tmp_path, sc)

    result = _run_cli("fit", str(path), "--period", str(sc.period), "--epoch", str(sc.epoch))
    assert result.returncode == 0
    assert "planet" in result.stdout


def test_cli_fit_json_output_is_clean_and_valid(tmp_path):
    sc = make_planet_curve()
    path = _write_npz(tmp_path, sc)

    result = _run_cli(
        "fit", str(path), "--period", str(sc.period), "--epoch", str(sc.epoch), "--json"
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)  # must parse cleanly: nothing else on stdout
    assert payload["verdict"] == "clear"
    assert payload["winner"] == "planet"


def test_cli_fit_no_significant_signal_exit_4(tmp_path):
    sc = make_null_curve()
    path = _write_npz(tmp_path, sc)

    result = _run_cli("fit", str(path), "--period", str(sc.period), "--epoch", str(sc.epoch))
    assert result.returncode == 4
    assert "no_significant_signal" in result.stdout


def test_cli_fit_missing_period_exit_2(tmp_path):
    sc = make_planet_curve()
    path = _write_npz(tmp_path, sc)

    result = _run_cli("fit", str(path))
    assert result.returncode == 2
    assert "no period available" in result.stderr


def test_cli_fit_bad_file_exit_2(tmp_path):
    bad_path = tmp_path / "nope.npz"
    result = _run_cli("fit", str(bad_path), "--period", "3.0")
    assert result.returncode == 2


def test_cli_fit_ambiguous_exit_3(tmp_path, monkeypatch):
    import fitr.cli as cli_module
    from fitr.compare import Comparison

    sc = make_planet_curve()
    path = _write_npz(tmp_path, sc)

    def fake_compare(results, phase, flux, flux_err):
        return Comparison(
            results=results,
            baseline_chi2=1000.0,
            baseline_bic=1000.0,
            delta_bic={r.model_name: 0.5 for r in results},
            winner=None,
            verdict="ambiguous",
            tied_models=["planet", "blend"],
            notes=["planet/blend degenerate: needs centroid vetting"],
        )

    monkeypatch.setattr(cli_module, "compare", fake_compare)
    exit_code = cli_module.main(
        ["fit", str(path), "--period", str(sc.period), "--epoch", str(sc.epoch)]
    )
    assert exit_code == 3


def test_cli_fit_uses_metadata_period_when_omitted(tmp_path):
    sc = make_planet_curve()
    path = tmp_path / "lc_meta.npz"
    np.savez(
        path,
        time=sc.lc.time,
        flux=sc.lc.flux,
        flux_err=sc.lc.flux_err,
        period_days=np.array(sc.period),
        epoch_btjd=np.array(sc.epoch),
    )

    result = _run_cli("fit", str(path))
    assert result.returncode == 0
    assert "period_days" in result.stderr
