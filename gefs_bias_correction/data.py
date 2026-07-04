from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


REQUIRED_DATA_VARS = ["forecast_t2m_c", "analysis_t2m_c"]
REQUIRED_COORDS = ["init_date", "fhr", "valid_time", "latitude", "longitude"]


def require_variables(ds: xr.Dataset) -> None:
    """Check that the matched GEFS/GFS dataset has required variables and coordinates."""
    missing_vars = [v for v in REQUIRED_DATA_VARS if v not in ds.data_vars]
    missing_coords = [c for c in REQUIRED_COORDS if c not in ds.coords]
    if missing_vars or missing_coords:
        raise ValueError(
            f"Dataset is missing variables {missing_vars} or coordinates {missing_coords}."
        )


def load_matched_dataset(path: str | Path, dtype: str = "float32") -> xr.Dataset:
    """Load a saved matched GEFS forecast and GFS analysis NetCDF dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find NetCDF file: {path}\n"
            "Update DATASET_PATH to the correct saved file location."
        )

    ds = xr.open_dataset(path).load()
    require_variables(ds)

    ds["forecast_t2m_c"] = ds["forecast_t2m_c"].astype(dtype)
    ds["analysis_t2m_c"] = ds["analysis_t2m_c"].astype(dtype)
    return ds


def print_dataset_summary(ds: xr.Dataset) -> None:
    """Print a concise dataset summary."""
    print("Dataset summary")
    print("---------------")
    print(f"Cases: {ds.sizes['case']}")
    print(f"Latitude points: {ds.sizes['latitude']}")
    print(f"Longitude points: {ds.sizes['longitude']}")
    print(f"Unique init dates: {len(np.unique(ds['init_date'].values))}")
    print(f"Forecast hours: {np.unique(ds['fhr'].values)}")
    print(
        "Forecast min/max: "
        f"{float(ds['forecast_t2m_c'].min()):.2f}, "
        f"{float(ds['forecast_t2m_c'].max()):.2f} °C"
    )
    print(
        "Analysis min/max: "
        f"{float(ds['analysis_t2m_c'].min()):.2f}, "
        f"{float(ds['analysis_t2m_c'].max()):.2f} °C"
    )
