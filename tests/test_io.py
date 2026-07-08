from __future__ import annotations

import numpy as np
import pytest

from fitr.io import FitrInputError, load_lightcurve


def test_load_npz_basic(tmp_path):
    time = np.linspace(0, 10, 100)
    flux = np.full(100, 2.0)
    flux_err = np.full(100, 0.01)
    path = tmp_path / "lc.npz"
    np.savez(path, time=time, flux=flux, flux_err=flux_err)

    lc = load_lightcurve(str(path))
    assert len(lc.time) == 100
    np.testing.assert_allclose(np.median(lc.flux), 1.0)


def test_load_npz_with_meta(tmp_path):
    time = np.linspace(0, 10, 50)
    flux = np.ones(50)
    flux_err = np.full(50, 0.01)
    path = tmp_path / "lc.npz"
    np.savez(
        path,
        time=time,
        flux=flux,
        flux_err=flux_err,
        period_days=np.array(3.14),
        epoch_btjd=np.array(1.5),
        tic_id=np.array(12345),
    )

    lc = load_lightcurve(str(path))
    assert lc.meta["period_days"] == pytest.approx(3.14)
    assert lc.meta["epoch_btjd"] == pytest.approx(1.5)
    assert lc.meta["tic_id"] == 12345


def test_load_npz_missing_required_array(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, time=np.arange(10), flux=np.ones(10))  # no flux_err

    with pytest.raises(FitrInputError, match="flux_err"):
        load_lightcurve(str(path))


def test_load_npz_length_mismatch(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, time=np.arange(10), flux=np.ones(9), flux_err=np.ones(10))

    with pytest.raises(FitrInputError, match="mismatched lengths"):
        load_lightcurve(str(path))


def test_load_npz_all_nan_flux(tmp_path):
    path = tmp_path / "bad.npz"
    n = 10
    np.savez(
        path,
        time=np.arange(n, dtype=float),
        flux=np.full(n, np.nan),
        flux_err=np.ones(n),
    )

    with pytest.raises(FitrInputError):
        load_lightcurve(str(path))


def test_load_npz_drops_nans(tmp_path):
    time = np.arange(10, dtype=float)
    flux = np.ones(10)
    flux[3] = np.nan
    flux_err = np.ones(10)
    path = tmp_path / "lc.npz"
    np.savez(path, time=time, flux=flux, flux_err=flux_err)

    lc = load_lightcurve(str(path))
    assert len(lc.time) == 9
    assert lc.meta["n_dropped"] == 1


def test_load_csv_with_flux_err(tmp_path):
    path = tmp_path / "lc.csv"
    path.write_text("time,flux,flux_err\n0.0,2.0,0.1\n1.0,2.0,0.1\n2.0,1.8,0.1\n")

    lc = load_lightcurve(str(path))
    assert len(lc.time) == 3
    assert "flux_err_estimated" not in lc.meta


def test_load_csv_without_flux_err(tmp_path):
    path = tmp_path / "lc.csv"
    rows = "\n".join(f"{i}.0,{1.0 + 0.001 * (i % 2)}" for i in range(20))
    path.write_text("time,flux\n" + rows + "\n")

    lc = load_lightcurve(str(path))
    assert lc.meta["flux_err_estimated"] is True
    assert np.all(lc.flux_err > 0)


def test_load_csv_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n")

    with pytest.raises(FitrInputError):
        load_lightcurve(str(path))


def test_load_nonexistent_file(tmp_path):
    with pytest.raises(FitrInputError):
        load_lightcurve(str(tmp_path / "does_not_exist.npz"))
