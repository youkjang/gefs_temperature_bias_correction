from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def plot_rmse_summary(summary: pd.DataFrame, save_path: str | Path | None = None):
    """Plot raw vs corrected RMSE and RMSE improvement."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)
    axes[0].plot(summary["fhr"], summary["rmse_c_raw"], marker="o", label="Raw GEFS ensemble mean")
    axes[0].plot(summary["fhr"], summary["rmse_c_corrected"], marker="o", label="Bias corrected")
    axes[0].set_title("Raw vs bias-corrected GEFS T2M RMSE")
    axes[0].set_xlabel("Forecast hour")
    axes[0].set_ylabel("Area-weighted RMSE (°C)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].axhline(0.0, linewidth=1)
    axes[1].plot(summary["fhr"], summary["rmse_improvement_c"], marker="o")
    axes[1].set_title("Positive values mean the correction improved RMSE")
    axes[1].set_xlabel("Forecast hour")
    axes[1].set_ylabel("RMSE improvement (°C)")
    axes[1].grid(True, alpha=0.3)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, axes


def plot_spatial_map(da: xr.DataArray, title: str, save_path: str | Path | None = None, cmap: str = "coolwarm"):
    """Plot a simple latitude-longitude map without Cartopy."""
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    vmax = float(np.nanpercentile(np.abs(da.values), 98))
    if vmax == 0 or not np.isfinite(vmax):
        vmax = None

    da.plot(
        ax=ax, x="longitude", y="latitude", cmap=cmap,
        vmin=-vmax if vmax else None, vmax=vmax,
        cbar_kwargs={"label": da.attrs.get("units", "")},
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.25)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_rmse_by_fhr(results: pd.DataFrame) -> None:
    """Plot RMSE by forecast hour for all methods."""
    plt.figure(figsize=(9, 5))
    for method, group in results.groupby("method"):
        group = group.sort_values("fhr")
        plt.plot(group["fhr"], group["rmse_c"], marker="o", label=method)

    plt.xlabel("Forecast hour")
    plt.ylabel("Area-weighted RMSE (°C)")
    plt.title("GEFS T2M RMSE: raw vs bias-correction methods")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_rmse_improvement(results_with_improvement: pd.DataFrame) -> None:
    """Plot RMSE improvement relative to raw GEFS."""
    df = results_with_improvement[results_with_improvement["method"] != "raw_gefs"].copy()

    plt.figure(figsize=(9, 5))
    for method, group in df.groupby("method"):
        group = group.sort_values("fhr")
        plt.plot(group["fhr"], group["rmse_improvement_c"], marker="o", label=method)

    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Forecast hour")
    plt.ylabel("RMSE improvement relative to raw GEFS (°C)")
    plt.title("Positive values mean lower RMSE than raw GEFS")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_spatial_error_example(
    ds: xr.Dataset,
    corrected_dict: dict[str, xr.DataArray],
    case_index: int = 0,
    vmin: float = -5,
    vmax: float = 5,
) -> None:
    """Plot spatial error maps for raw and selected correction methods for one test case."""
    methods = {"raw_gefs": ds["forecast_t2m_c"]}
    methods.update(corrected_dict)

    n_methods = len(methods)
    fig, axes = plt.subplots(
        1,
        n_methods,
        figsize=(5 * n_methods, 4),
        constrained_layout=True,
    )
    if n_methods == 1:
        axes = [axes]

    fhr = int(ds["fhr"].isel(case=case_index).item())
    valid_time = pd.to_datetime(ds["valid_time"].isel(case=case_index).values).strftime(
        "%Y-%m-%d"
    )

    for ax, (name, da) in zip(axes, methods.items()):
        error = da.isel(case=case_index) - ds["analysis_t2m_c"].isel(case=case_index)
        im = ax.pcolormesh(
            ds["longitude"],
            ds["latitude"],
            error,
            shading="auto",
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"{name}\nf{fhr:03d}, valid {valid_time}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.colorbar(im, ax=ax, label="Error (°C)")

    plt.show()

def plot_rmse_and_improvement(results_df):
    """Plot RMSE and RMSE improvement by forecast lead time."""
    methods = list(results_df["method"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    for method in methods:
        sub = results_df[results_df["method"] == method].sort_values("fhr")
        axes[0].plot(sub["fhr"], sub["rmse_c"], marker="o", label=method)

    axes[0].set_title("GEFS T2M RMSE: raw vs spatial-feature bias-correction methods")
    axes[0].set_ylabel("Area-weighted RMSE (°C)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for method in methods:
        if method == "raw_gefs":
            continue
        sub = results_df[results_df["method"] == method].sort_values("fhr")
        axes[1].plot(sub["fhr"], sub["rmse_improvement_c"], marker="o", label=method)

    axes[1].axhline(0, linewidth=1)
    axes[1].set_title("Positive values mean lower RMSE than raw GEFS")
    axes[1].set_xlabel("Forecast hour")
    axes[1].set_ylabel("RMSE improvement relative to raw GEFS (°C)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    plt.show()
