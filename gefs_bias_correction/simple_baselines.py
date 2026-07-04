"""Simple baseline corrections used for comparison with ML."""

from __future__ import annotations

import numpy as np
import xarray as xr


def split_train_test_by_init_date(ds: xr.Dataset, train_fraction: float = 0.7) -> tuple[xr.Dataset, xr.Dataset]:
    """Split a case dataset by unique initialization date."""
    dates = np.asarray(ds["init_date"].values)
    unique_dates = np.unique(dates)
    n_train = int(np.floor(len(unique_dates) * train_fraction))
    train_dates = unique_dates[:n_train]

    train_mask = np.isin(dates, train_dates)
    test_mask = ~train_mask

    return ds.isel(case=train_mask), ds.isel(case=test_mask)


def compute_mean_bias_by_lead(
    train_ds: xr.Dataset,
    forecast_var: str = "forecast_t2m_c",
    analysis_var: str = "analysis_t2m_c",
) -> xr.DataArray:
    """Compute lead-time-dependent mean bias: forecast - analysis."""
    fields = []
    fhrs = sorted(np.unique(train_ds["fhr"].values).astype(int))
    for fhr in fhrs:
        subset = train_ds.where(train_ds["fhr"] == fhr, drop=True)
        bias = (subset[forecast_var] - subset[analysis_var]).mean("case")
        bias = bias.expand_dims(fhr=[int(fhr)])
        fields.append(bias)
    return xr.concat(fields, dim="fhr").rename("mean_bias_c")


def apply_mean_bias_correction(
    ds: xr.Dataset,
    mean_bias_by_lead: xr.DataArray,
    forecast_var: str = "forecast_t2m_c",
) -> xr.DataArray:
    """Apply lead-time-dependent mean-bias correction."""
    corrected_cases = []
    for i in range(ds.sizes["case"]):
        fhr = int(ds["fhr"].isel(case=i).values)
        corrected = ds[forecast_var].isel(case=i) - mean_bias_by_lead.sel(fhr=fhr)
        corrected = corrected.expand_dims(case=[ds["case"].values[i]])
        corrected_cases.append(corrected)
    out = xr.concat(corrected_cases, dim="case")
    out = out.assign_coords(fhr=ds["fhr"], init_date=ds["init_date"])
    return out.rename("mean_bias_corrected_t2m_c")
