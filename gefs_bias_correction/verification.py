import numpy as np
import pandas as pd
import xarray as xr


def area_weights(da: xr.DataArray, lat_name: str = "latitude") -> xr.DataArray:
    """Return cosine-latitude area weights."""
    weights = np.cos(np.deg2rad(da[lat_name]))
    return weights / weights.mean()

def weighted_mean_2d(da: xr.DataArray) -> xr.DataArray:
    """Compute an area-weighted spatial mean over latitude and longitude."""
    weights = area_weights(da["latitude"])
    return da.weighted(weights).mean(dim=("latitude", "longitude"))

def area_weighted_mean(da: xr.DataArray) -> xr.DataArray:
    """Area-weighted mean over latitude and longitude."""
    weights = area_weights(da)
    return da.weighted(weights).mean(dim=("latitude", "longitude"))


def summarize_by_lead(forecast: xr.DataArray, analysis: xr.DataArray, fhr_coord: xr.DataArray, label: str) -> pd.DataFrame:
    """Summarize area-weighted bias, RMSE, and MAE by forecast lead time."""
    rows = []
    for fhr in sorted(np.unique(fhr_coord.values)):
        mask = fhr_coord == fhr
        fcst_fhr = forecast.where(mask, drop=True)
        anal_fhr = analysis.where(mask, drop=True)
        err = fcst_fhr - anal_fhr
        mse_map = (err ** 2).mean("case")
        mae_map = np.abs(err).mean("case")
        bias_map = err.mean("case")
        rows.append({
            "method": label,
            "fhr": int(fhr),
            "bias_c": float(area_weighted_mean(bias_map)),
            "rmse_c": float(np.sqrt(area_weighted_mean(mse_map))),
            "mae_c": float(area_weighted_mean(mae_map)),
            "n_cases": int(fcst_fhr.sizes.get("case", 0)),
        })
    return pd.DataFrame(rows)


def make_summary_table(test_ds: xr.Dataset, corrected: xr.DataArray) -> pd.DataFrame:
    """Create raw-vs-corrected summary table by forecast lead time."""
    raw = summarize_by_lead(test_ds["forecast_t2m_c"], test_ds["analysis_t2m_c"], test_ds["fhr"], label="raw")
    corr = summarize_by_lead(corrected, test_ds["analysis_t2m_c"], test_ds["fhr"], label="corrected")
    wide = raw.merge(corr, on="fhr", suffixes=("_raw", "_corrected"))
    wide["rmse_improvement_c"] = wide["rmse_c_raw"] - wide["rmse_c_corrected"]
    wide["mae_improvement_c"] = wide["mae_c_raw"] - wide["mae_c_corrected"]
    wide["abs_bias_improvement_c"] = np.abs(wide["bias_c_raw"]) - np.abs(wide["bias_c_corrected"])
    columns = [
        "fhr", "n_cases_raw",
        "bias_c_raw", "bias_c_corrected", "abs_bias_improvement_c",
        "rmse_c_raw", "rmse_c_corrected", "rmse_improvement_c",
        "mae_c_raw", "mae_c_corrected", "mae_improvement_c",
    ]
    return wide[columns].sort_values("fhr").reset_index(drop=True)

def compute_metrics_by_fhr(
    ds: xr.Dataset,
    forecast_da: xr.DataArray,
    method_name: str,
) -> pd.DataFrame:
    """Compute area-weighted bias, RMSE, and MAE by forecast hour."""
    rows: list[dict[str, float | int | str]] = []
    analysis = ds["analysis_t2m_c"]

    for fhr in sorted(np.unique(ds["fhr"].values)):
        mask = ds["fhr"] == fhr
        forecast = forecast_da.where(mask, drop=True)
        observed = analysis.where(mask, drop=True)
        error = forecast - observed

        bias_cases = weighted_mean_2d(error)
        rmse_cases = np.sqrt(weighted_mean_2d(error**2))
        mae_cases = weighted_mean_2d(np.abs(error))

        rows.append(
            {
                "method": method_name,
                "fhr": int(fhr),
                "bias_c": float(bias_cases.mean()),
                "rmse_c": float(rmse_cases.mean()),
                "mae_c": float(mae_cases.mean()),
                "n_cases": int(mask.sum()),
            }
        )

    return pd.DataFrame(rows)

def combine_results_with_improvement(results: pd.DataFrame) -> pd.DataFrame:
    """Add RMSE, MAE, and absolute-bias improvement relative to raw GEFS."""
    raw = results[results["method"] == "raw_gefs"][["fhr", "rmse_c", "mae_c", "bias_c"]]
    raw = raw.rename(
        columns={
            "rmse_c": "raw_rmse_c",
            "mae_c": "raw_mae_c",
            "bias_c": "raw_bias_c",
        }
    )

    out = results.merge(raw, on="fhr", how="left")
    out["rmse_improvement_c"] = out["raw_rmse_c"] - out["rmse_c"]
    out["mae_improvement_c"] = out["raw_mae_c"] - out["mae_c"]
    out["abs_bias_improvement_c"] = np.abs(out["raw_bias_c"]) - np.abs(out["bias_c"])
    return out.sort_values(["fhr", "method"]).reset_index(drop=True)

def verify_forecast_by_lead(ds_input, forecast_da, method_name: str) -> pd.DataFrame:
    """Compute area-weighted bias, RMSE, and MAE by forecast lead time."""
    rows = []
    fhrs = sorted(np.unique(ds_input["fhr"].values))

    for fhr in fhrs:
        mask = ds_input["fhr"] == fhr
        obs = ds_input["analysis_t2m_c"].where(mask, drop=True)
        fcst = forecast_da.where(mask, drop=True)
        error = fcst - obs

        bias = float(area_weighted_mean(error).values)
        rmse = float(np.sqrt(area_weighted_mean(error**2)).values)
        mae = float(area_weighted_mean(abs(error)).values)
        n_cases = int(obs.sizes["case"])
        rows.append(
            {
                "method": method_name,
                "fhr": int(fhr),
                "bias_c": bias,
                "rmse_c": rmse,
                "mae_c": mae,
                "n_cases": n_cases,
            }
        )
    return pd.DataFrame(rows)

def build_verification_table(test_ds, mean_bias_corrected, ml_corrected_dict: dict) -> pd.DataFrame:
    """Build comparison table for raw GEFS, mean-bias correction, and ML methods."""
    raw_da = test_ds["forecast_t2m_c"]
    tables = [
        verify_forecast_by_lead(test_ds, raw_da, "raw_gefs"),
        verify_forecast_by_lead(test_ds, mean_bias_corrected, "mean_bias"),
    ]
    for name, da in ml_corrected_dict.items():
        tables.append(verify_forecast_by_lead(test_ds, da, name))

    results = pd.concat(tables, ignore_index=True)
    raw_ref = (
        results[results["method"] == "raw_gefs"][["fhr", "rmse_c", "mae_c", "bias_c"]]
        .rename(columns={"rmse_c": "raw_rmse_c", "mae_c": "raw_mae_c", "bias_c": "raw_bias_c"})
    )
    results = results.merge(raw_ref, on="fhr", how="left")
    results["rmse_improvement_c"] = results["raw_rmse_c"] - results["rmse_c"]
    results["mae_improvement_c"] = results["raw_mae_c"] - results["mae_c"]
    results["abs_bias_improvement_c"] = abs(results["raw_bias_c"]) - abs(results["bias_c"])
    return results.sort_values(["fhr", "method"]).reset_index(drop=True)

