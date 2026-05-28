import numpy as np
import h5py
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import os

# ══════════════════════════════════════════════════════════════════════════════
# 10. THERMAL ANOMALY & Z-SCORE (universal)
# ══════════════════════════════════════════════════════════════════════════════

def compute_thermal_anomaly(lst_kelvin: np.ndarray,
                            background_method: str = "mean",
                            background_value: float = None) -> dict:
    """
    Hitung Thermal Anomaly (ΔT) dan Z-score dari array LST.

        ΔT = T_pixel - T_background
        Z  = (T - μ) / σ

    Universal — bisa menerima input dari band M maupun I.

    Parameters
    ----------
    lst_kelvin        : np.ndarray — LST dalam Kelvin (lst_m15_m16_kelvin atau lst_i5_kelvin)
    background_method : str        — 'mean' | 'median' | 'custom'
    background_value  : float      — nilai background (K) jika method='custom'

    Returns
    -------
    dict:
        'lst_celsius'    : LST dalam °C
        'delta_t'        : Thermal anomaly ΔT (K)
        'z_score'        : Z-score per piksel
        'background_k'   : nilai background yang dipakai (K)
        'stats'          : dict ringkasan statistik (min, max, mean, std, hot_pixels)
    """
    lst_celsius = lst_kelvin - 273.15

    if background_method == "mean":
        bg_k = np.nanmean(lst_kelvin)
    elif background_method == "median":
        bg_k = np.nanmedian(lst_kelvin)
    elif background_method == "custom":
        if background_value is None:
            raise ValueError("background_value wajib diisi jika method='custom'")
        bg_k = background_value
    else:
        raise ValueError(f"background_method tidak dikenali: '{background_method}'")

    delta_t = lst_kelvin - bg_k

    mu    = np.nanmean(lst_kelvin)
    sigma = np.nanstd(lst_kelvin)
    with np.errstate(divide="ignore", invalid="ignore"):
        z_score = (lst_kelvin - mu) / (sigma + 1e-10)

    stats = {
        # Key baru (v2)
        "lst_min_celsius" : float(np.nanmin(lst_celsius)),
        "lst_max_celsius" : float(np.nanmax(lst_celsius)),
        "lst_mean_celsius": float(np.nanmean(lst_celsius)),
        "lst_std_celsius" : float(np.nanstd(lst_celsius)),
        "background_k"    : float(bg_k),
        "delta_t_max"     : float(np.nanmax(delta_t)),
        "z_score_max"     : float(np.nanmax(z_score)),
        "hot_pixels"      : int(np.sum(z_score > 2.0)),
    }
    # Backward-compat alias — key lama (v1) tetap ada, point ke value yang sama
    stats["lst_min_c"]  = stats["lst_min_celsius"]
    stats["lst_max_c"]  = stats["lst_max_celsius"]
    stats["lst_mean_c"] = stats["lst_mean_celsius"]
    stats["lst_std_c"]  = stats["lst_std_celsius"]

    return {
        "lst_celsius" : lst_celsius,
        "delta_t"     : delta_t,
        "z_score"     : z_score,
        "background_k": bg_k,
        "stats"       : stats,
    }

