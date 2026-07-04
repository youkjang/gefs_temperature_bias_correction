from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

FEATURE_NAMES = np.array(
    [
        "forecast_t2m_c",
        "fhr",
        "latitude",
        "longitude",
        "valid_month",
        "valid_day_of_year",
    ]
)


def make_feature_arrays(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Convert gridded xarray data into ML feature and target arrays.

    Target is the correction term: analysis_t2m_c - forecast_t2m_c.
    """
    forecast = ds["forecast_t2m_c"].values.astype("float32")
    analysis = ds["analysis_t2m_c"].values.astype("float32")
    n_case, n_lat, n_lon = forecast.shape

    lat_vals = ds["latitude"].values.astype("float32")
    lon_vals = ds["longitude"].values.astype("float32")
    fhr_vals = ds["fhr"].values.astype("float32")
    valid_times = pd.to_datetime(ds["valid_time"].values)
    month_vals = valid_times.month.astype("float32")
    doy_vals = valid_times.dayofyear.astype("float32")

    raw_feature = forecast.reshape(-1)
    fhr_feature = np.repeat(fhr_vals, n_lat * n_lon)
    month_feature = np.repeat(month_vals, n_lat * n_lon)
    doy_feature = np.repeat(doy_vals, n_lat * n_lon)
    lat_feature = np.tile(np.repeat(lat_vals, n_lon), n_case)
    lon_feature = np.tile(np.tile(lon_vals, n_lat), n_case)
    target = (analysis - forecast).reshape(-1)

    X = np.column_stack(
        [raw_feature, fhr_feature, lat_feature, lon_feature, month_feature, doy_feature]
    ).astype("float32")

    valid = np.isfinite(X).all(axis=1) & np.isfinite(target)
    X = X[valid]
    y = target[valid].astype("float32")

    feature_info = {
        "feature_names": FEATURE_NAMES,
        "valid_mask": valid,
        "original_shape": np.array([n_case, n_lat, n_lon]),
    }
    return X, y, feature_info


def sample_training_data(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int = 300_000,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly sample training rows to keep ML fitting manageable."""
    n_samples = X.shape[0]
    if n_samples <= max_samples:
        return X, y

    rng = np.random.default_rng(random_seed)
    idx = rng.choice(n_samples, size=max_samples, replace=False)
    return X[idx], y[idx]


def make_features_for_prediction(ds_subset: xr.Dataset) -> np.ndarray:
    """Create feature array for prediction for a dataset subset."""
    forecast = ds_subset["forecast_t2m_c"].values.astype("float32")
    n_case, n_lat, n_lon = forecast.shape

    lat_vals = ds_subset["latitude"].values.astype("float32")
    lon_vals = ds_subset["longitude"].values.astype("float32")
    fhr_vals = ds_subset["fhr"].values.astype("float32")
    valid_times = pd.to_datetime(ds_subset["valid_time"].values)
    month_vals = valid_times.month.astype("float32")
    doy_vals = valid_times.dayofyear.astype("float32")

    raw_feature = forecast.reshape(-1)
    fhr_feature = np.repeat(fhr_vals, n_lat * n_lon)
    month_feature = np.repeat(month_vals, n_lat * n_lon)
    doy_feature = np.repeat(doy_vals, n_lat * n_lon)
    lat_feature = np.tile(np.repeat(lat_vals, n_lon), n_case)
    lon_feature = np.tile(np.tile(lon_vals, n_lat), n_case)

    return np.column_stack(
        [raw_feature, fhr_feature, lat_feature, lon_feature, month_feature, doy_feature]
    ).astype("float32")
