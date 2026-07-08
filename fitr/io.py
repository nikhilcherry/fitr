"""Light curve loading: arvyo .npz schema (v1.0) and plain CSV."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class FitrInputError(Exception):
    """Raised when an input light curve file is malformed or unreadable."""


@dataclass
class LightCurve:
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


_REQUIRED_NPZ_ARRAYS = ("time", "flux", "flux_err")
_OPTIONAL_NPZ_ARRAYS = ("flux_raw",)
_NPZ_META_KEYS = (
    "tic_id",
    "label",
    "sector",
    "period_days",
    "epoch_btjd",
    "crowdsap",
    "mission",
)


def load_lightcurve(path: str) -> LightCurve:
    """Load a light curve from an arvyo .npz file or a plain CSV file."""
    if str(path).lower().endswith(".npz"):
        return _load_npz(path)
    return _load_csv(path)


def _load_npz(path: str) -> LightCurve:
    try:
        data = np.load(path, allow_pickle=True)
    except OSError as exc:
        raise FitrInputError(f"could not read npz file '{path}': {exc}") from exc

    for key in _REQUIRED_NPZ_ARRAYS:
        if key not in data.files:
            raise FitrInputError(
                f"npz file '{path}' is missing required array '{key}' "
                f"(arvyo schema v1.0 requires: {', '.join(_REQUIRED_NPZ_ARRAYS)})"
            )

    time = np.asarray(data["time"], dtype=float)
    flux = np.asarray(data["flux"], dtype=float)
    flux_err = np.asarray(data["flux_err"], dtype=float)

    for name, arr in (("time", time), ("flux", flux), ("flux_err", flux_err)):
        if arr.ndim != 1:
            raise FitrInputError(
                f"npz array '{name}' in '{path}' must be 1-D, got shape {arr.shape}"
            )

    if not (len(time) == len(flux) == len(flux_err)):
        raise FitrInputError(
            f"npz arrays in '{path}' have mismatched lengths: "
            f"time={len(time)}, flux={len(flux)}, flux_err={len(flux_err)}"
        )

    if len(flux) == 0 or not np.any(np.isfinite(flux)):
        raise FitrInputError(f"npz file '{path}' has no finite flux values")

    meta: dict[str, Any] = {}
    if "flux_raw" in data.files:
        meta["flux_raw"] = np.asarray(data["flux_raw"], dtype=float)

    if "meta" in data.files:
        raw_meta = data["meta"]
        try:
            raw_meta = raw_meta.item()
        except (AttributeError, ValueError):
            pass
        if isinstance(raw_meta, dict):
            for key in _NPZ_META_KEYS:
                if key in raw_meta:
                    meta[key] = raw_meta[key]

    for key in _NPZ_META_KEYS:
        if key in data.files and key not in meta:
            value = data[key]
            try:
                value = value.item()
            except (AttributeError, ValueError):
                pass
            meta[key] = value

    return _finalize(time, flux, flux_err, meta)


def _load_csv(path: str) -> LightCurve:
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise FitrInputError(f"CSV file '{path}' has no header row")
            fieldnames = [name.strip() for name in reader.fieldnames]
            if "time" not in fieldnames or "flux" not in fieldnames:
                raise FitrInputError(
                    f"CSV file '{path}' must have columns 'time,flux[,flux_err]', "
                    f"found: {', '.join(fieldnames)}"
                )
            has_err = "flux_err" in fieldnames
            times, fluxes, errs = [], [], []
            for row in reader:
                times.append(_to_float(row.get("time")))
                fluxes.append(_to_float(row.get("flux")))
                errs.append(_to_float(row.get("flux_err")) if has_err else np.nan)
    except OSError as exc:
        raise FitrInputError(f"could not read CSV file '{path}': {exc}") from exc

    time = np.asarray(times, dtype=float)
    flux = np.asarray(fluxes, dtype=float)

    if len(flux) == 0 or not np.any(np.isfinite(flux)):
        raise FitrInputError(f"CSV file '{path}' has no finite flux values")

    meta: dict[str, Any] = {}
    if has_err:
        flux_err = np.asarray(errs, dtype=float)
    else:
        flux_err = _estimate_flux_err(flux)
        meta["flux_err_estimated"] = True

    return _finalize(time, flux, flux_err, meta)


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def _estimate_flux_err(flux: np.ndarray) -> np.ndarray:
    diffs = np.diff(flux[np.isfinite(flux)])
    mad = np.median(np.abs(diffs - np.median(diffs))) if len(diffs) else 0.0
    sigma = 1.4826 * mad / np.sqrt(2)
    if not np.isfinite(sigma) or sigma == 0.0:
        sigma = 1.0
    return np.full(flux.shape, sigma, dtype=float)


def _finalize(
    time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, meta: dict[str, Any]
) -> LightCurve:
    finite_mask = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    n_dropped = int(np.size(finite_mask) - np.count_nonzero(finite_mask))
    time = time[finite_mask]
    flux = flux[finite_mask]
    flux_err = flux_err[finite_mask]

    if len(flux) == 0:
        raise FitrInputError("no finite data points remain after dropping NaN/inf")

    median_flux = np.median(flux)
    if median_flux != 0:
        flux = flux / median_flux
        flux_err = flux_err / median_flux

    meta["n_dropped"] = n_dropped

    return LightCurve(time=time, flux=flux, flux_err=flux_err, meta=meta)
