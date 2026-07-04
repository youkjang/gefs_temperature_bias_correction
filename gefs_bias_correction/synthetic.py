"""Small synthetic dataset for testing the ML notebook without downloading GEFS data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def make_synthetic_matched_dataset(
    start_date: str = "2023-06-01",
    end_date: str = "2023-08-31",
    forecast_hours: list[int] | None = None,
    n_lat: int = 20,
    n_lon: int = 30,
    seed: int = 42,
) -> xr.Dataset:
    """Create a synthetic GEFS/GFS matched dataset with realistic-looking T2M structure."""
    if forecast_hours is None:
        forecast_hours = [24, 48, 72, 96, 120]

    rng = np.random.default_rng(seed)
    init_dates = pd.date_range(start_date, end_date, freq="D")
    lat = np.linspace(25.0, 42.0, n_lat)
    lon = np.linspace(-107.0, -90.0, n_lon)

    cases = []
    fhrs = []
    dates = []
    for d in init_dates:
        for fhr in forecast_hours:
            cases.append(f"{d.strftime('%Y%m%d')}_f{fhr:03d}")
            fhrs.append(fhr)
            dates.append(d)

    n_case = len(cases)
    lon2d, lat2d = np.meshgrid(lon, lat)

    forecast = np.empty((n_case, n_lat, n_lon), dtype=float)
    analysis = np.empty_like(forecast)

    for i, (date, fhr) in enumerate(zip(dates, fhrs)):
        doy = date.dayofyear
        seasonal = 30 + 3.0 * np.sin((doy - 172) / 365 * 2 * np.pi)
        spatial = -0.18 * (lat2d - 33) + 0.04 * (lon2d + 100)
        synoptic = rng.normal(0, 1.0)
        true_field = seasonal + spatial + synoptic + rng.normal(0, 0.7, size=(n_lat, n_lon))

        # Forecast bias depends on lead time and location.
        warm_bias = 0.3 + 0.002 * fhr + 0.08 * (lat2d - lat2d.mean())
        error_noise = rng.normal(0, 0.6 + 0.003 * fhr, size=(n_lat, n_lon))
        analysis[i] = true_field
        forecast[i] = true_field + warm_bias + error_noise

    ds = xr.Dataset(
        {
            "forecast_t2m_c": (("case", "latitude", "longitude"), forecast),
            "analysis_t2m_c": (("case", "latitude", "longitude"), analysis),
        },
        coords={
            "case": cases,
            "latitude": lat,
            "longitude": lon,
            "fhr": ("case", np.asarray(fhrs, dtype=int)),
            "init_date": ("case", np.asarray(dates, dtype="datetime64[ns]")),
        },
        attrs={"description": "Synthetic matched GEFS forecast and GFS analysis T2M dataset"},
    )
    return ds
