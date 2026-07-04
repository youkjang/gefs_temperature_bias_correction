"""
Feature-table utilities for GEFS 2-m temperature machine-learning bias correction.

The ML target is a correction term:
    target_c = analysis_t2m_c - forecast_t2m_c

A trained model predicts target_c, and the corrected forecast is:
    ml_corrected_t2m_c = forecast_t2m_c + predicted_correction_c
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_FEATURE_COLUMNS = [
    "forecast_t2m_c",
    "fhr",
    "latitude",
    "longitude",
    "month",
    "day_of_year",
]


@dataclass(frozen=True)
class FeatureTableMetadata:
    """Shape and coordinate metadata needed to reconstruct gridded predictions."""

    shape: tuple[int, int, int]
    dims: tuple[str, str, str]
    latitude_name: str
    longitude_name: str


def _find_lat_lon_names(ds: xr.Dataset | xr.DataArray) -> tuple[str, str]:
    """Return likely latitude and longitude coordinate names."""
    lat_candidates = ["latitude", "lat", "y"]
    lon_candidates = ["longitude", "lon", "x"]

    lat_name = next((name for name in lat_candidates if name in ds.coords or name in ds.dims), None)
    lon_name = next((name for name in lon_candidates if name in ds.coords or name in ds.dims), None)

    if lat_name is None or lon_name is None:
        raise ValueError(
            "Could not identify latitude/longitude names. Expected coordinates such as "
            "'latitude'/'longitude' or 'lat'/'lon'."
        )
    return lat_name, lon_name


def _case_datetime_values(ds: xr.Dataset) -> pd.DatetimeIndex:
    """Return initialization dates as pandas datetimes."""
    if "init_date" in ds.coords:
        values = ds["init_date"].values
    elif "init_date" in ds:
        values = ds["init_date"].values
    else:
        raise ValueError("Dataset must contain an 'init_date' coordinate or variable on the case dimension.")

    return pd.to_datetime(values)


def _case_fhr_values(ds: xr.Dataset) -> np.ndarray:
    """Return forecast-hour values on the case dimension."""
    if "fhr" in ds.coords:
        return np.asarray(ds["fhr"].values).astype(int)
    if "fhr" in ds:
        return np.asarray(ds["fhr"].values).astype(int)
    raise ValueError("Dataset must contain an 'fhr' coordinate or variable on the case dimension.")


def make_ml_feature_table(
    ds: xr.Dataset,
    forecast_var: str = "forecast_t2m_c",
    analysis_var: str = "analysis_t2m_c",
    include_target: bool = True,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, FeatureTableMetadata]:
    """
    Convert a gridded case/lat/lon dataset into a scikit-learn feature table.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with dimensions (case, latitude, longitude) and variables
        forecast_t2m_c and analysis_t2m_c.
    forecast_var : str
        Forecast variable name.
    analysis_var : str
        Analysis/verification variable name.
    include_target : bool
        If True, add target_c = analysis - forecast.
    sample_size : int or None
        Optional random sample size. Use this only for training if memory is limited.
        Do not sample test data when reconstructing gridded predictions.
    random_state : int
        Random seed for sampling.

    Returns
    -------
    df : pandas.DataFrame
        Feature table with one row per finite gridpoint/case value.
    metadata : FeatureTableMetadata
        Metadata needed to reconstruct full gridded predictions.
    """
    if forecast_var not in ds:
        raise ValueError(f"Dataset does not contain required forecast variable: {forecast_var}")
    if include_target and analysis_var not in ds:
        raise ValueError(f"Dataset does not contain required analysis variable: {analysis_var}")

    da = ds[forecast_var]
    if da.ndim != 3:
        raise ValueError(
            f"{forecast_var} must be 3D with dimensions like (case, latitude, longitude). "
            f"Got dimensions: {da.dims}"
        )

    dims = da.dims
    case_dim = dims[0]
    lat_name, lon_name = _find_lat_lon_names(ds)

    if case_dim != "case":
        raise ValueError(f"Expected first dimension to be 'case'. Got {case_dim!r}.")

    forecast = np.asarray(ds[forecast_var].values)
    shape = forecast.shape
    n_case, n_lat, n_lon = shape

    case_idx, lat_idx, lon_idx = np.indices(shape)
    case_idx = case_idx.ravel()
    lat_idx = lat_idx.ravel()
    lon_idx = lon_idx.ravel()

    forecast_flat = forecast.ravel()

    lat_values = np.asarray(ds[lat_name].values)
    lon_values = np.asarray(ds[lon_name].values)
    fhr_values = _case_fhr_values(ds)
    init_dates = _case_datetime_values(ds)

    df = pd.DataFrame(
        {
            "case_index": case_idx,
            "lat_index": lat_idx,
            "lon_index": lon_idx,
            "forecast_t2m_c": forecast_flat,
            "latitude": lat_values[lat_idx],
            "longitude": lon_values[lon_idx],
            "fhr": fhr_values[case_idx],
            "init_date": init_dates[case_idx],
        }
    )

    df["month"] = df["init_date"].dt.month.astype(int)
    df["day_of_year"] = df["init_date"].dt.dayofyear.astype(int)

    finite_mask = np.isfinite(df["forecast_t2m_c"].values)

    if include_target:
        analysis_flat = np.asarray(ds[analysis_var].values).ravel()
        df["analysis_t2m_c"] = analysis_flat
        df["target_c"] = df["analysis_t2m_c"] - df["forecast_t2m_c"]
        finite_mask &= np.isfinite(df["analysis_t2m_c"].values)
        finite_mask &= np.isfinite(df["target_c"].values)

    df = df.loc[finite_mask].reset_index(drop=True)

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    metadata = FeatureTableMetadata(
        shape=shape,
        dims=tuple(dims),
        latitude_name=lat_name,
        longitude_name=lon_name,
    )
    return df, metadata


def feature_target_arrays(
    df: pd.DataFrame,
    feature_columns: Iterable[str] = DEFAULT_FEATURE_COLUMNS,
    target_column: str = "target_c",
) -> tuple[pd.DataFrame, pd.Series]:
    """Return X and y arrays for scikit-learn training."""
    feature_columns = list(feature_columns)
    missing = [col for col in feature_columns + [target_column] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[feature_columns].astype(float)
    y = df[target_column].astype(float)
    return X, y


def reconstruct_corrected_dataarray(
    ds: xr.Dataset,
    prediction_df: pd.DataFrame,
    predicted_correction_column: str = "predicted_correction_c",
    forecast_var: str = "forecast_t2m_c",
    name: str = "ml_corrected_t2m_c",
) -> xr.DataArray:
    """
    Reconstruct a corrected forecast DataArray from predicted correction values.

    prediction_df must include case_index, lat_index, lon_index, and predicted_correction_c.
    """
    required = ["case_index", "lat_index", "lon_index", predicted_correction_column]
    missing = [col for col in required if col not in prediction_df.columns]
    if missing:
        raise ValueError(f"prediction_df is missing required columns: {missing}")

    raw = np.asarray(ds[forecast_var].values)
    corrected = np.full(raw.shape, np.nan, dtype=float)

    case_idx = prediction_df["case_index"].to_numpy(dtype=int)
    lat_idx = prediction_df["lat_index"].to_numpy(dtype=int)
    lon_idx = prediction_df["lon_index"].to_numpy(dtype=int)
    correction = prediction_df[predicted_correction_column].to_numpy(dtype=float)

    corrected[case_idx, lat_idx, lon_idx] = raw[case_idx, lat_idx, lon_idx] + correction

    out = xr.DataArray(
        corrected,
        dims=ds[forecast_var].dims,
        coords=ds[forecast_var].coords,
        name=name,
        attrs={
            "units": "degC",
            "description": "GEFS 2-m temperature corrected with machine-learning-predicted correction term",
        },
    )
    return out
