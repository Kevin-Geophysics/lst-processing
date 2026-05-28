
import numpy as np
import h5py
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import os

# ══════════════════════════════════════════════════════════════════════════════
# 3. DN → SPECTRAL RADIANCE (universal M & I)
# ══════════════════════════════════════════════════════════════════════════════

def dn_to_radiance(dn: np.ndarray,
                   scale_factor: float,
                   add_offset: float,
                   band: str = "") -> np.ndarray:
    """
    Konversi Digital Number (DN) ke Spectral Radiance.

        L_λ = DN × scale_factor + add_offset

    Universal untuk semua band VIIRS thermal (M15, M16, I5).
    Scale factor dan offset diambil dari metadata file HDF5 masing-masing granule.

    Parameters
    ----------
    dn           : np.ndarray — DN raw dari load_viirs_bands()
    scale_factor : float      — multiplicative scaling factor (dari metadata granule)
    add_offset   : float      — additive offset (dari metadata granule)
    band         : str        — nama band untuk logging, misal 'M15', 'M16', 'I5'

    Returns
    -------
    np.ndarray : Spectral radiance (W·m⁻²·sr⁻¹·µm⁻¹)

    Referensi: USGS Landsat Collection 2 Product Guide; VNP02IMG User Guide NASA LP DAAC
    """
    radiance = dn * scale_factor + add_offset
    radiance = np.where(radiance > 0, radiance, np.nan)
    return radiance


# ══════════════════════════════════════════════════════════════════════════════
# 4. SPECTRAL RADIANCE → BRIGHTNESS TEMPERATURE (universal M & I)
# ══════════════════════════════════════════════════════════════════════════════

def radiance_to_bt(radiance: np.ndarray,
                   k1: float,
                   k2: float,
                   band: str = "") -> np.ndarray:
    """
    Konversi Spectral Radiance ke Brightness Temperature via Planck inverse.

        BT = K2 / ln(K1 / L_λ + 1)

    Universal untuk semua band VIIRS thermal. Gunakan K1/K2 sesuai band:
        Band M15 : K1=865.8087,  K2=1292.8520  (NASA VIIRS Calibration Support Team)
        Band M16 : K1=596.8099,  K2=1195.0224
        Band I5  : K1=620.004,   K2=1258.74

    Parameters
    ----------
    radiance : np.ndarray — spectral radiance hasil dn_to_radiance()
    k1       : float      — thermal conversion constant 1
    k2       : float      — thermal conversion constant 2
    band     : str        — nama band untuk logging, misal 'M15', 'M16', 'I5'

    Returns
    -------
    np.ndarray : Brightness Temperature (K)

    Referensi: Wan & Dozier (1996); Jiménez-Muñoz et al. (2014)
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        bt = k2 / np.log((k1 / radiance) + 1)

    bt = np.where(np.isfinite(bt) & (bt > 150) & (bt < 400), bt, np.nan)
    return bt


# ══════════════════════════════════════════════════════════════════════════════
# 5. HITUNG NDVI
# ══════════════════════════════════════════════════════════════════════════════

def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Hitung NDVI dari band Red dan NIR.

        NDVI = (NIR - Red) / (NIR + Red)

    Parameters
    ----------
    red : np.ndarray — reflektansi band Red (surface reflectance, bukan DN)
                       VIIRS: band I1 (0.64µm) atau M5 (0.672µm)
    nir : np.ndarray — reflektansi band NIR
                       VIIRS: band I2 (0.86µm) atau M7 (0.865µm)

    Returns
    -------
    np.ndarray : NDVI [-1, 1]
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red + 1e-10)
    ndvi = np.clip(ndvi, -1.0, 1.0)
    return ndvi


# ══════════════════════════════════════════════════════════════════════════════
# 6. HITUNG EMISSIVITY (universal M & I, via param band)
# ══════════════════════════════════════════════════════════════════════════════

def compute_emissivity(ndvi: np.ndarray,
                       band: str,
                       ndvi_soil: float = 0.2,
                       ndvi_veg: float  = 0.5) -> dict:
    """
    Estimasi emissivity permukaan dari NDVI untuk band M atau I.

    Dua metode tersedia tergantung parameter `band`:

    band='M' → Proporsi Vegetasi (Wan 2014):
        Pv      = ((NDVI - NDVI_min) / (NDVI_max - NDVI_min))²
        ε_M15   = 0.9869 - 0.0029 × Pv
        ε_M16   = 0.9894 - 0.0026 × Pv
        ε_mean  = (ε_M15 + ε_M16) / 2
        Δε      = ε_M15 - ε_M16

    band='I' → NDVI Threshold Method (Sobrino et al. 2008):
        NDVI < ndvi_soil → ε_I5 = 0.9720 (tanah murni)
        NDVI > ndvi_veg  → ε_I5 = 0.9900 (vegetasi penuh)
        Di antara keduanya → interpolasi dengan cavity effect correction:
            Pv    = ((NDVI - ndvi_soil) / (ndvi_veg - ndvi_soil))²
            ε_I5  = 0.9900 × Pv + 0.9720 × (1 - Pv) + 4e-4 × Pv × (1 - Pv)

    Parameters
    ----------
    ndvi      : np.ndarray — array NDVI dari compute_ndvi()
    band      : str        — 'M' untuk band M15/M16, 'I' untuk band I5
    ndvi_soil : float      — threshold NDVI tanah murni (default 0.2)
    ndvi_veg  : float      — threshold NDVI vegetasi penuh (default 0.5)

    Returns
    -------
    dict — untuk band='M' (semua key tersedia):
        'emissivity_m15'   : np.ndarray — emissivity band M15 per piksel [~0.983–0.987]
        'emissivity_m16'   : np.ndarray — emissivity band M16 per piksel [~0.986–0.989]
        'emissivity_mean'  : np.ndarray — (emissivity_m15 + emissivity_m16) / 2
                             → dipakai sebagai input split_window_lst()
        'delta_emissivity' : np.ndarray — emissivity_m15 - emissivity_m16
                             → dipakai sebagai input split_window_lst()
        'pv'               : np.ndarray — proporsi vegetasi [0, 1]

    dict — untuk band='I' (subset — emissivity_mean dan delta_emissivity TIDAK ada):
        'emissivity_i5'    : np.ndarray — emissivity band I5 per piksel [~0.972–0.990]
                             → dipakai sebagai input single_channel_lst()
        'pv'               : np.ndarray — proporsi vegetasi [0, 1]

    Catatan: emissivity_mean dan delta_emissivity hanya ada di output band='M'
             karena Split-Window membutuhkan dua band. Single Channel I5 hanya
             butuh satu nilai emissivity (emissivity_i5).

    Referensi: Wan (2014), RSE 140; Sobrino et al. (2008), RSE; Jiménez-Muñoz et al. (2014)
    """
    band = band.upper()

    if band == "M":
        # Proporsi vegetasi dari persentil 5–95 untuk robustness
        ndvi_min = np.nanpercentile(ndvi, 5)
        ndvi_max = np.nanpercentile(ndvi, 95)
        pv = ((ndvi - ndvi_min) / (ndvi_max - ndvi_min + 1e-10)) ** 2
        pv = np.clip(pv, 0.0, 1.0)

        emissivity_m15 = EMISS_M15_A + EMISS_M15_B * pv   # 0.9869 - 0.0029 × Pv
        emissivity_m16 = EMISS_M16_A + EMISS_M16_B * pv   # 0.9894 - 0.0026 × Pv

        return {
            "emissivity_m15"   : emissivity_m15,
            "emissivity_m16"   : emissivity_m16,
            "emissivity_mean"  : (emissivity_m15 + emissivity_m16) / 2,
            "delta_emissivity" : emissivity_m15 - emissivity_m16,
            "pv"               : pv,
        }

    elif band == "I":
        pv = ((ndvi - ndvi_soil) / (ndvi_veg - ndvi_soil + 1e-10)) ** 2
        pv = np.clip(pv, 0.0, 1.0)

        # Emissivity campuran dengan cavity effect correction untuk piksel mixed
        e_mixed = (EMISS_I5_VEG * pv
                   + EMISS_I5_SOIL * (1 - pv)
                   + 4e-4 * pv * (1 - pv))

        emissivity_i5 = np.where(ndvi < ndvi_soil, EMISS_I5_SOIL,
                        np.where(ndvi > ndvi_veg,  EMISS_I5_VEG,
                                 e_mixed))
        emissivity_i5 = np.where(np.isfinite(ndvi), emissivity_i5, np.nan)

        return {
            "emissivity_i5": emissivity_i5,
            "pv"           : pv,
        }

    else:
        raise ValueError(f"Parameter band harus 'M' atau 'I', bukan '{band}'")


# ══════════════════════════════════════════════════════════════════════════════
# 7. GET WATER VAPOR
# ══════════════════════════════════════════════════════════════════════════════

def get_water_vapor(source: str = "constant",
                    wv_file: str = None,
                    wv_constant: float = 2.5,
                    shape: tuple = None) -> np.ndarray:
    """
    Ambil nilai water vapor (g/cm²) untuk dipakai di Split-Window Algorithm.

    Parameters
    ----------
    source      : str   — 'constant' | 'file'
                  'constant' → pakai nilai seragam (cocok untuk preliminary study)
                  'file'     → load dari GeoTIFF (ERA5 / MERRA-2 hasil export GEE)
    wv_file     : str   — path ke GeoTIFF water vapor (wajib jika source='file')
    wv_constant : float — nilai WV konstan (g/cm²), default 2.5 untuk tropis
    shape       : tuple — (rows, cols) wajib diisi jika source='constant'

    Returns
    -------
    np.ndarray : Water vapor (g/cm²)

    Catatan:
        Wilayah tropis (Indonesia) umumnya WV = 2.0–4.5 g/cm².
        Untuk preliminary study, 2.5 g/cm² sudah representatif.
        Untuk akurasi lebih tinggi, export WV dari:
            GEE → ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
                  band: 'total_column_water_vapour' (konversi: × 0.1)
    """
    if source == "constant":
        if shape is None:
            raise ValueError("Jika source='constant', parameter 'shape' wajib diisi.")
        wv = np.full(shape, wv_constant, dtype=np.float64)
        print(f"💧 Water vapor : konstan {wv_constant} g/cm² (tropis default)")
        return wv

    elif source == "file":
        if wv_file is None:
            raise ValueError("Jika source='file', parameter 'wv_file' wajib diisi.")
        with rasterio.open(wv_file) as src:
            wv = src.read(1).astype(np.float64)
        wv = np.where(wv > 0, wv, np.nan)
        print(f"💧 Water vapor : loaded dari {os.path.basename(wv_file)}")
        print(f"   Range       : {np.nanmin(wv):.2f} – {np.nanmax(wv):.2f} g/cm²")
        return wv

    else:
        raise ValueError(f"source harus 'constant' atau 'file', bukan '{source}'")


# ══════════════════════════════════════════════════════════════════════════════
# 8. SPLIT-WINDOW LST (khusus Band M15 + M16)
# ══════════════════════════════════════════════════════════════════════════════

def split_window_lst(bt_m15_kelvin: np.ndarray,
                     bt_m16_kelvin: np.ndarray,
                     emissivity_mean: np.ndarray,
                     delta_emissivity: np.ndarray,
                     water_vapor: np.ndarray,
                     coeff: dict = None) -> np.ndarray:
    """
    Hitung LST menggunakan Split-Window Algorithm dari band M15 dan M16.

        LST = T_M15 + c1(T_M15 - T_M16) + c2(T_M15 - T_M16)²
              + c0 + (c3 + c4·w)(1 - ε) + (c5 + c6·w)·Δε

    Parameters
    ----------
    bt_m15_kelvin    : np.ndarray — Brightness Temperature M15 (K)
                                    hasil radiance_to_bt(..., band='M15')
    bt_m16_kelvin    : np.ndarray — Brightness Temperature M16 (K)
                                    hasil radiance_to_bt(..., band='M16')
    emissivity_mean  : np.ndarray — mean emissivity (M15+M16)/2
                                    dari compute_emissivity(band='M')['emissivity_mean']
    delta_emissivity : np.ndarray — selisih emissivity M15-M16
                                    dari compute_emissivity(band='M')['delta_emissivity']
    water_vapor      : np.ndarray — water vapor g/cm² dari get_water_vapor()
    coeff            : dict       — override koefisien SW (default: SW_COEFF global)

    Returns
    -------
    np.ndarray : LST dalam Kelvin (K) — simpan sebagai lst_m15_m16_kelvin

    Referensi: Wan & Dozier (1996), IEEE TGRS 34(4); Jiménez-Muñoz et al. (2014)
    """
    c = coeff if coeff is not None else SW_COEFF

    diff = bt_m15_kelvin - bt_m16_kelvin
    lst_m15_m16_kelvin = (bt_m15_kelvin
                          + c["c1"] * diff
                          + c["c2"] * diff ** 2
                          + c["c0"]
                          + (c["c3"] + c["c4"] * water_vapor) * (1 - emissivity_mean)
                          + (c["c5"] + c["c6"] * water_vapor) * delta_emissivity)

    lst_m15_m16_kelvin = np.where(
        (lst_m15_m16_kelvin > 200) & (lst_m15_m16_kelvin < 400),
        lst_m15_m16_kelvin, np.nan
    )
    return lst_m15_m16_kelvin


# ══════════════════════════════════════════════════════════════════════════════
# 9. SINGLE CHANNEL LST (universal M15 atau I5, via param wavelength)
# ══════════════════════════════════════════════════════════════════════════════

def single_channel_lst(bt_kelvin: np.ndarray,
                       emissivity: np.ndarray,
                       wavelength: float,
                       band: str = "") -> np.ndarray:
    """
    Hitung LST menggunakan Single Channel Algorithm.

        LST = BT / (1 + (λ × BT / ρ) × ln ε)

    Universal untuk band M15 maupun I5 — bedanya hanya parameter wavelength
    dan emissivity yang dipass.

    Untuk Band M15:
        wavelength = LAMBDA_M15  (10.763e-6 m)
        emissivity = compute_emissivity(band='M')['emissivity_m15']

    Untuk Band I5:
        wavelength = LAMBDA_I5   (11.450e-6 m)
        emissivity = compute_emissivity(band='I')['emissivity_i5']

    Parameters
    ----------
    bt_kelvin  : np.ndarray — Brightness Temperature (K) hasil radiance_to_bt()
    emissivity : np.ndarray — surface emissivity sesuai band
    wavelength : float      — panjang gelombang band (m), gunakan LAMBDA_M15 atau LAMBDA_I5
    band       : str        — nama band untuk logging, misal 'M15' atau 'I5'

    Returns
    -------
    np.ndarray : LST dalam Kelvin (K)
                 Simpan sebagai lst_m15_kelvin (jika band M15)
                 atau lst_i5_kelvin (jika band I5)

    Catatan: Single Channel kurang akurat untuk WV > 3 g/cm² (kondisi tropis lembab).
             Untuk band M, gunakan Split-Window sebagai metode utama.
             Single Channel I5 adalah satu-satunya pilihan untuk data VNP02IMG.

    Referensi: Jiménez-Muñoz et al. (2014), IEEE GRSL 11(10), 1840–1843
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        log_emiss = np.log(np.where(emissivity > 0, emissivity, np.nan))
        lst_kelvin = bt_kelvin / (1 + (wavelength * bt_kelvin / RHO) * log_emiss)

    lst_kelvin = np.where(
        (lst_kelvin > 200) & (lst_kelvin < 400),
        lst_kelvin, np.nan
    )
    return lst_kelvin
