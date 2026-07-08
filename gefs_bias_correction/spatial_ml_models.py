from __future__ import annotations

import xarray as xr

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .spatial_features import make_spatial_feature_table


def train_spatial_ml_models(
    train_df,
    feature_columns,
    target_column,
    random_seed: int = 42,
):
    """Train spatial-feature ML models for GEFS T2M bias correction."""
    X = train_df[feature_columns]
    y = train_df[target_column]

    models = {
        "linear_spatial": make_pipeline(StandardScaler(), LinearRegression()),
        "ridge_spatial": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "hist_gradient_boosting_spatial": HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.10,
            random_state=random_seed,
        ),
    }

    fitted = {}
    for name, model in models.items():
        print(f"Training {name}...")
        fitted[name] = model.fit(X, y)
    return fitted


def predict_corrected_forecast(
    model,
    ds_input,
    feature_columns,
    case_chunk: int = 10,
    verbose: bool = False,
) -> xr.DataArray:
    """Predict correction in chunks and return ML-corrected forecast as xarray.DataArray."""
    corrected_chunks = []
    n_cases = ds_input.sizes["case"]

    for start in range(0, n_cases, case_chunk):
        end = min(start + case_chunk, n_cases)
        if verbose:
            print(f"Predicting cases {start}:{end}")

        subset = ds_input.isel(case=slice(start, end))
        feature_df = make_spatial_feature_table(subset, include_target=False)
        pred_correction = model.predict(feature_df[feature_columns]).astype("float32")
        pred_correction_3d = pred_correction.reshape(
            subset.sizes["case"], subset.sizes["latitude"], subset.sizes["longitude"]
        )

        corrected_values = subset["forecast_t2m_c"].values.astype("float32") + pred_correction_3d
        corrected_da = xr.DataArray(
            corrected_values,
            dims=("case", "latitude", "longitude"),
            coords={
                "case": subset["case"].values,
                "latitude": subset["latitude"].values,
                "longitude": subset["longitude"].values,
                "init_date": ("case", subset["init_date"].values),
                "fhr": ("case", subset["fhr"].values),
                "valid_time": ("case", subset["valid_time"].values),
            },
            name="ml_corrected_t2m_c",
            attrs={"units": "degC"},
        )
        corrected_chunks.append(corrected_da)

    return xr.concat(corrected_chunks, dim="case")


def apply_models_to_test_dataset(models: dict, test_ds, feature_columns, case_chunk=10, verbose=False):
    """Apply each fitted model to the test dataset."""
    corrected = {}
    for name, model in models.items():
        print(f"Applying {name} correction...")
        corrected[name] = predict_corrected_forecast(
            model,
            test_ds,
            feature_columns,
            case_chunk=case_chunk,
            verbose=verbose,
        )
    return corrected
