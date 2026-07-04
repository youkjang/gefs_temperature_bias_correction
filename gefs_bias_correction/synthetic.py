import numpy as np
import pandas as pd
import xarray as xr

from .config import ProjectConfig


def make_synthetic_t2m_dataset(config: ProjectConfig, seed: int = 7) -> xr.Dataset:
    """Create a synthetic matched forecast-analysis dataset for debugging."""
    rng = np.random.default_rng(seed)

    lats = np.arange(config.region.south, config.region.north + 0.1, 2.5)
    lons = np.arange(config.region.west, config.region.east + 0.1, 2.5)
    lon2d, lat2d = np.meshgrid(lons, lats)

    forecast_cases = []
    analysis_cases = []
    init_values = []
    fhr_values = []
    valid_values = []

    for init_date in config.init_dates:
        init_dt = pd.to_datetime(f"{init_date} {config.init_hour}:00")
        for fhr in config.forecast_hours:
            valid_dt = init_dt + pd.Timedelta(hours=int(fhr))

            seasonal = 8.0 + 12.0 * np.cos(np.deg2rad(lat2d - 30.0))
            wave = 2.0 * np.sin(np.deg2rad(lon2d + 100.0))
            lead_bias = 0.2 + 0.015 * int(fhr)
            spatial_bias = 0.8 * np.sin(np.deg2rad(lat2d * 2.0))
            noise_analysis = rng.normal(0.0, 0.8, size=lat2d.shape)
            noise_forecast = rng.normal(0.0, 1.0, size=lat2d.shape)

            analysis = seasonal + wave + noise_analysis
            forecast = analysis + lead_bias + spatial_bias + noise_forecast

            forecast_da = xr.DataArray(
                forecast,
                dims=("latitude", "longitude"),
                coords={"latitude": lats, "longitude": lons},
                name="forecast_t2m_c",
                attrs={"units": "degC"},
            )
            analysis_da = xr.DataArray(
                analysis,
                dims=("latitude", "longitude"),
                coords={"latitude": lats, "longitude": lons},
                name="analysis_t2m_c",
                attrs={"units": "degC"},
            )

            case_id = len(forecast_cases)
            forecast_cases.append(forecast_da.expand_dims(case=[case_id]))
            analysis_cases.append(analysis_da.expand_dims(case=[case_id]))
            init_values.append(init_dt.normalize())
            fhr_values.append(int(fhr))
            valid_values.append(valid_dt)

    forecast_all = xr.concat(forecast_cases, dim="case", compat="override", coords="minimal", combine_attrs="override")
    analysis_all = xr.concat(analysis_cases, dim="case", compat="override", coords="minimal", combine_attrs="override")

    ds = xr.Dataset({"forecast_t2m_c": forecast_all, "analysis_t2m_c": analysis_all})
    ds = ds.assign_coords(
        init_date=("case", pd.to_datetime(init_values)),
        fhr=("case", np.asarray(fhr_values, dtype=int)),
        valid_time=("case", pd.to_datetime(valid_values)),
    )
    return ds
