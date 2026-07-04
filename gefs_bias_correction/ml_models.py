"""Machine-learning models and verification utilities for GEFS T2M bias correction."""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


MODEL_FEATURE_COLUMNS = [
    "forecast_t2m_c",
    "fhr",
    "latitude",
    "longitude",
    "month",
    "day_of_year",
]


def make_default_ml_models(random_state: int = 42) -> OrderedDict:
    """
    Return a small set of baseline ML models for correction-term prediction.

    The target should be analysis_t2m_c - forecast_t2m_c.
    """
    return OrderedDict(
        {
            "linear_regression": make_pipeline(StandardScaler(), LinearRegression()),
            "random_forest": RandomForestRegressor(
                n_estimators=50,
                max_depth=14,
                min_samples_leaf=5,
                n_jobs=1,
                random_state=random_state,
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                random_state=random_state,
            ),
        }
    )


def train_ml_models(
    train_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_column: str = "target_c",
    random_state: int = 42,
    verbose: bool = False,
) -> OrderedDict:
    """Train default ML correction models and return fitted estimators."""
    if feature_columns is None:
        feature_columns = MODEL_FEATURE_COLUMNS

    X_train = train_df[feature_columns].astype(float)
    y_train = train_df[target_column].astype(float)

    models = make_default_ml_models(random_state=random_state)
    fitted = OrderedDict()

    for name, model in models.items():
        if verbose:
            print(f"Training {name} with {len(train_df):,} samples...")
        model.fit(X_train, y_train)
        fitted[name] = model

    return fitted


def predict_corrections(
    models: Mapping[str, object],
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Predict correction terms for each model and return dataframes with metadata columns."""
    if feature_columns is None:
        feature_columns = MODEL_FEATURE_COLUMNS

    X = df[feature_columns].astype(float)
    out: dict[str, pd.DataFrame] = {}

    keep_cols = ["case_index", "lat_index", "lon_index", "forecast_t2m_c", "fhr", "init_date"]
    if "analysis_t2m_c" in df.columns:
        keep_cols.append("analysis_t2m_c")
    if "target_c" in df.columns:
        keep_cols.append("target_c")

    for name, model in models.items():
        pred = model.predict(X)
        pred_df = df[keep_cols].copy()
        pred_df["predicted_correction_c"] = pred
        pred_df["ml_corrected_t2m_c"] = pred_df["forecast_t2m_c"] + pred_df["predicted_correction_c"]
        out[name] = pred_df

    return out


def correction_skill_on_table(
    prediction_df: pd.DataFrame,
    target_column: str = "target_c",
    predicted_column: str = "predicted_correction_c",
) -> dict[str, float]:
    """Evaluate predicted correction terms against true correction terms in tabular form."""
    y_true = prediction_df[target_column].to_numpy(dtype=float)
    y_pred = prediction_df[predicted_column].to_numpy(dtype=float)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    return {"correction_rmse_c": float(rmse), "correction_mae_c": float(mae), "correction_bias_c": bias}


def _latitude_weights(lat_values: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat_values)).astype(float)
    weights = np.where(np.isfinite(weights), weights, np.nan)
    return weights


def _weighted_spatial_case_mean(values: xr.DataArray, lat_name: str) -> xr.DataArray:
    weights = xr.DataArray(
        _latitude_weights(values[lat_name].values),
        dims=(lat_name,),
        coords={lat_name: values[lat_name]},
    )
    spatial_dims = [dim for dim in values.dims if dim != "case"]
    return values.weighted(weights).mean(dim=spatial_dims)


def verification_by_lead(
    ds_test: xr.Dataset,
    corrected_methods: Mapping[str, xr.DataArray],
    forecast_var: str = "forecast_t2m_c",
    analysis_var: str = "analysis_t2m_c",
) -> pd.DataFrame:
    """
    Compute area-weighted bias, RMSE, and MAE by forecast lead time.

    Parameters
    ----------
    ds_test : xarray.Dataset
        Independent test dataset.
    corrected_methods : mapping
        Dictionary of method name -> corrected forecast DataArray.
        Example: {"mean_bias": corrected_mean_bias, "random_forest": rf_corrected}
    """
    lat_name = "latitude" if "latitude" in ds_test.coords else "lat"
    methods: dict[str, xr.DataArray] = {"raw": ds_test[forecast_var]}
    methods.update(corrected_methods)

    rows = []
    fhrs = sorted(np.unique(ds_test["fhr"].values).astype(int))

    for fhr in fhrs:
        mask = ds_test["fhr"] == fhr
        obs = ds_test[analysis_var].where(mask, drop=True)
        n_cases = int(obs.sizes.get("case", 0))

        for method_name, forecast_da in methods.items():
            fcst = forecast_da.where(mask, drop=True)
            err = fcst - obs

            bias_cases = _weighted_spatial_case_mean(err, lat_name)
            rmse_cases = _weighted_spatial_case_mean(err ** 2, lat_name)
            mae_cases = _weighted_spatial_case_mean(np.abs(err), lat_name)

            rows.append(
                {
                    "method": method_name,
                    "fhr": int(fhr),
                    "bias_c": float(bias_cases.mean("case").values),
                    "rmse_c": float(np.sqrt(rmse_cases.mean("case").values)),
                    "mae_c": float(mae_cases.mean("case").values),
                    "n_cases": n_cases,
                }
            )

    out = pd.DataFrame(rows)

    raw = out[out["method"] == "raw"][["fhr", "rmse_c", "mae_c", "bias_c"]].rename(
        columns={"rmse_c": "raw_rmse_c", "mae_c": "raw_mae_c", "bias_c": "raw_bias_c"}
    )
    out = out.merge(raw, on="fhr", how="left")
    out["rmse_improvement_c"] = out["raw_rmse_c"] - out["rmse_c"]
    out["mae_improvement_c"] = out["raw_mae_c"] - out["mae_c"]
    out["abs_bias_improvement_c"] = np.abs(out["raw_bias_c"]) - np.abs(out["bias_c"])

    return out
