from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

FEATURE_COLUMNS = [
    "forecast_t2m_c",
    "fhr",
    "fhr_norm",
    "latitude",
    "longitude",
    "lat_norm",
    "lon_norm",
    "lat_norm2",
    "lon_norm2",
    "lat_lon_interaction",
    "forecast_x_fhr",
    "forecast_x_lat",
    "forecast_x_lon",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
]

TARGET_COLUMN = "target_correction_c"


def _datetime_features(valid_time_values) -> dict[str, np.ndarray]:
    """Return cyclic month and day-of-year features from valid_time."""
    dt = pd.to_datetime(valid_time_values)
    month = dt.month.to_numpy(dtype="float32")
    doy = dt.dayofyear.to_numpy(dtype="float32")

    month_angle = 2.0 * np.pi * (month - 1.0) / 12.0
    doy_angle = 2.0 * np.pi * (doy - 1.0) / 366.0

    return {
        "month_sin": np.sin(month_angle).astype("float32"),
        "month_cos": np.cos(month_angle).astype("float32"),
        "doy_sin": np.sin(doy_angle).astype("float32"),
        "doy_cos": np.cos(doy_angle).astype("float32"),
    }


def make_spatial_feature_table(ds_input: xr.Dataset, include_target: bool = True) -> pd.DataFrame:
    """Convert gridded xarray data into a pandas DataFrame for scikit-learn.

    Each row represents one grid point at one forecast case.
    """
    forecast = ds_input["forecast_t2m_c"].values.astype("float32")
    n_case, n_lat, n_lon = forecast.shape

    lats = ds_input["latitude"].values.astype("float32")
    lons = ds_input["longitude"].values.astype("float32")
    lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")

    lat_norm_2d = ((lat2d - np.nanmean(lats)) / np.nanstd(lats)).astype("float32")
    lon_norm_2d = ((lon2d - np.nanmean(lons)) / np.nanstd(lons)).astype("float32")

    fhr = ds_input["fhr"].values.astype("float32")
    fhr_std = np.nanstd(fhr)
    if fhr_std == 0:
        fhr_norm = np.zeros_like(fhr, dtype="float32")
    else:
        fhr_norm = ((fhr - np.nanmean(fhr)) / fhr_std).astype("float32")

    time_features = _datetime_features(ds_input["valid_time"].values)

    raw_flat = forecast.reshape(-1)
    lat_flat = np.broadcast_to(lat2d, (n_case, n_lat, n_lon)).reshape(-1)
    lon_flat = np.broadcast_to(lon2d, (n_case, n_lat, n_lon)).reshape(-1)
    latn_flat = np.broadcast_to(lat_norm_2d, (n_case, n_lat, n_lon)).reshape(-1)
    lonn_flat = np.broadcast_to(lon_norm_2d, (n_case, n_lat, n_lon)).reshape(-1)

    fhr_flat = np.repeat(fhr, n_lat * n_lon)
    fhrn_flat = np.repeat(fhr_norm, n_lat * n_lon)
    month_sin_flat = np.repeat(time_features["month_sin"], n_lat * n_lon)
    month_cos_flat = np.repeat(time_features["month_cos"], n_lat * n_lon)
    doy_sin_flat = np.repeat(time_features["doy_sin"], n_lat * n_lon)
    doy_cos_flat = np.repeat(time_features["doy_cos"], n_lat * n_lon)

    df = pd.DataFrame(
        {
            "forecast_t2m_c": raw_flat,
            "fhr": fhr_flat,
            "fhr_norm": fhrn_flat,
            "latitude": lat_flat,
            "longitude": lon_flat,
            "lat_norm": latn_flat,
            "lon_norm": lonn_flat,
            "lat_norm2": latn_flat**2,
            "lon_norm2": lonn_flat**2,
            "lat_lon_interaction": latn_flat * lonn_flat,
            "forecast_x_fhr": raw_flat * fhrn_flat,
            "forecast_x_lat": raw_flat * latn_flat,
            "forecast_x_lon": raw_flat * lonn_flat,
            "month_sin": month_sin_flat,
            "month_cos": month_cos_flat,
            "doy_sin": doy_sin_flat,
            "doy_cos": doy_cos_flat,
        }
    )

    if include_target:
        analysis = ds_input["analysis_t2m_c"].values.astype("float32")
        df[TARGET_COLUMN] = (analysis - forecast).reshape(-1)

    return df.replace([np.inf, -np.inf], np.nan).dropna()


def subsample_training_table(train_df: pd.DataFrame, max_samples: int, random_seed: int = 42) -> pd.DataFrame:
    """Subsample training rows to keep Colab memory use manageable."""
    if len(train_df) > max_samples:
        return train_df.sample(n=max_samples, random_state=random_seed).reset_index(drop=True)
    return train_df.reset_index(drop=True)
