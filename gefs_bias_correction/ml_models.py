from __future__ import annotations

import numpy as np
import xarray as xr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from .ml_features import make_features_for_prediction


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_seed: int = 42,
    run_random_forest: bool = False,
    verbose: bool = True,
) -> dict[str, object]:
    """Train baseline scikit-learn ML models for bias correction."""
    models: dict[str, object] = {}

    if verbose:
        print("Training Linear Regression...")
    models["linear_regression"] = LinearRegression()
    models["linear_regression"].fit(X_train, y_train)

    if verbose:
        print("Training Ridge Regression...")
    models["ridge"] = Ridge(alpha=1.0, random_state=random_seed)
    models["ridge"].fit(X_train, y_train)

    if verbose:
        print("Training HistGradientBoostingRegressor...")
    models["hist_gradient_boosting"] = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=31,
        random_state=random_seed,
    )
    models["hist_gradient_boosting"].fit(X_train, y_train)

    if run_random_forest:
        if verbose:
            print("Training RandomForestRegressor...")
        models["random_forest"] = RandomForestRegressor(
            n_estimators=80,
            max_depth=18,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=random_seed,
        )
        models["random_forest"].fit(X_train, y_train)

    return models


def predict_corrected_forecast(
    model: object,
    ds: xr.Dataset,
    model_name: str,
    case_chunk: int = 10,
    verbose: bool = False,
) -> xr.DataArray:
    """Predict correction term and return ML-corrected gridded forecast."""
    corrected_chunks = []
    n_case = ds.sizes["case"]

    for start in range(0, n_case, case_chunk):
        end = min(start + case_chunk, n_case)
        ds_chunk = ds.isel(case=slice(start, end))
        X_pred = make_features_for_prediction(ds_chunk)

        pred_correction = model.predict(X_pred).astype("float32")
        pred_correction = pred_correction.reshape(ds_chunk["forecast_t2m_c"].shape)

        corrected = ds_chunk["forecast_t2m_c"] + xr.DataArray(
            pred_correction,
            dims=ds_chunk["forecast_t2m_c"].dims,
            coords=ds_chunk["forecast_t2m_c"].coords,
        )
        corrected_chunks.append(corrected)

        if verbose:
            print(f"Predicted cases {start}:{end}")

    corrected_da = xr.concat(corrected_chunks, dim="case")
    corrected_da.name = f"{model_name}_corrected_t2m_c"
    corrected_da.attrs["units"] = "degC"
    return corrected_da
