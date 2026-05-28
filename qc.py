import numpy as np
import h5py
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import os

# ══════════════════════════════════════════════════════════════════════════════
# 11. APPLY QC MASK (universal)
# ══════════════════════════════════════════════════════════════════════════════

def apply_qc_mask(data: np.ndarray,
                  qc: np.ndarray,
                  good_flags: list = [0, 1]) -> np.ndarray:
    """
    Masking piksel buruk menggunakan QC / Quality Flag band VIIRS.

    QC bit 0-1 VNP21IMG (Band M):
        00 (0) = Pixel produced, good quality
        01 (1) = Pixel produced, unreliable quality
        10 (2) = Pixel not produced — cloud effects
        11 (3) = Pixel not produced — other reasons

    Untuk Band I (VNP02IMG), QF_I5 menggunakan skema bit berbeda —
    sesuaikan parameter good_flags jika diperlukan.

    Parameters
    ----------
    data       : np.ndarray — array LST atau BT yang akan dimasking
    qc         : np.ndarray — QC atau QF band dari load_viirs_bands()
    good_flags : list       — nilai flag yang dianggap valid (default [0, 1])

    Returns
    -------
    np.ndarray : data yang sudah dimasking (piksel buruk → NaN)

    Referensi: VNP21 User Guide, NASA LP DAAC
    """
    if qc is None:
        print("⚠️  QC band tidak tersedia, masking dilewati.")
        return data

    qc_2bit     = qc & 0b11
    mask_valid  = np.isin(qc_2bit, good_flags)
    masked      = np.where(mask_valid, data, np.nan)

    n_masked = int(np.sum(~mask_valid))
    n_total  = int(np.sum(np.isfinite(data)))
    print(f"🎭 QC mask : {n_masked} piksel dibuang dari {n_total} valid "
          f"({100 * n_masked / max(n_total, 1):.1f}%)")

    return masked
