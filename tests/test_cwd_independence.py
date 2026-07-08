"""Mandatory regression: `fitr fit` must work from any cwd with a relative
input path. This is the class of bug that has bitten sibling tools before
(paths silently resolved against the process cwd instead of the caller's)."""

from __future__ import annotations

import shutil
import subprocess
import sys

import numpy as np

from tests.conftest import make_planet_curve


def _write_fixture(foreign_dir):
    sc = make_planet_curve()
    data_dir = foreign_dir / "data"
    data_dir.mkdir()
    path = data_dir / "lc.npz"
    np.savez(path, time=sc.lc.time, flux=sc.lc.flux, flux_err=sc.lc.flux_err)
    return sc, "data/lc.npz"


def test_cwd_independence_via_module_invocation(tmp_path):
    foreign_dir = tmp_path / "foreign_cwd"
    foreign_dir.mkdir()
    sc, relative_path = _write_fixture(foreign_dir)

    result = subprocess.run(
        [
            sys.executable, "-m", "fitr.cli", "fit", relative_path,
            "--period", str(sc.period), "--epoch", str(sc.epoch),
        ],
        cwd=str(foreign_dir),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "planet" in result.stdout


def test_cwd_independence_via_installed_console_script(tmp_path):
    fitr_bin = shutil.which("fitr")
    if fitr_bin is None:
        import pytest

        pytest.skip("fitr console script not on PATH in this environment")

    foreign_dir = tmp_path / "foreign_cwd_console"
    foreign_dir.mkdir()
    sc, relative_path = _write_fixture(foreign_dir)

    result = subprocess.run(
        [fitr_bin, "fit", relative_path, "--period", str(sc.period), "--epoch", str(sc.epoch)],
        cwd=str(foreign_dir),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "planet" in result.stdout
