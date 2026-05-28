import numpy as np
import h5py
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import os

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD VIIRS BANDS (universal — auto-detect M15/M16 atau I5)
# ══════════════════════════════════════════════════════════════════════════════

def load_viirs_bands(hdf5_path: str, verbose: bool = True) -> dict:
    """
    Load band thermal VIIRS dari file HDF5 — auto-detect band yang tersedia.

    Mendukung dua jenis file:
        VNP21IMG / VJ121IMG → Band M (M15, M16, QC)
        VNP02IMG / VJ102IMG → Band I (I5, QF_I5)

    File akan di-scan otomatis — band yang ditemukan saja yang diload.
    Hasilnya bisa dicek dengan melihat key yang ada di dict return:
        'M15' ada → file Band M
        'I5'  ada → file Band I
        Keduanya ada → file mengandung kedua band

    Lat/lon TIDAK ada di file VNP02IMG — load terpisah via load_geo_i().
    Lat/lon di file VNP21IMG tersedia langsung di sini.

    Parameters
    ----------
    hdf5_path : str  — path ke file .h5 dari NASA Earthdata
    verbose   : bool — tampilkan info band yang ditemukan jika True

    Returns
    -------
    dict dengan keys (hanya key yang ditemukan yang ada):
        'M15'      : np.ndarray float64 — DN band M15 (jika ada)
        'M16'      : np.ndarray float64 — DN band M16 (jika ada)
        'I5'       : np.ndarray float64 — DN band I5  (jika ada)
        'QC'       : np.ndarray         — QC band M (jika ada)
        'QF_I5'    : np.ndarray         — Quality Flag I5 (jika ada)
        'lat'      : np.ndarray         — latitude per piksel (jika ada di file)
        'lon'      : np.ndarray         — longitude per piksel (jika ada di file)
        'lut_i5'   : np.ndarray float64 — BT LUT untuk I5 (jika ada, index = SI value)
        'metadata' : dict               — atribut global file
    """
    # Kandidat nama dataset per band di berbagai versi file NASA
    BAND_CANDIDATES = {
        "M15"   : ["M15", "EV_BandM15", "BrightnessTemperature_M15"],
        "M16"   : ["M16", "EV_BandM16", "BrightnessTemperature_M16"],
        "I5"    : ["I05", "I5", "EV_BandI5"],
        "I1"    : ["I01", "I1", "EV_BandI1"],
        "I2"    : ["I02", "I2", "EV_BandI2"],
        "QC"    : ["QC", "QC_Day", "QC_Night", "LST_QC"],
        "QF_I5" : ["I05_quality_flags", "I5_quality_flags"],
        "lut_i5": ["I05_brightness_temperature_lut", "I5_brightness_temperature_lut"],
        "lat"   : ["Latitude", "latitude", "lat"],
        "lon"   : ["Longitude", "longitude", "lon"],
        
    }

    data = {"metadata": {}}

    def _find(f, candidates):
        """Cari dataset dari daftar kandidat nama, cek root dan satu level sub-grup."""
        for key in candidates:
            if key in f:
                return f[key][:]
            for grp in f.values():
                if hasattr(grp, "keys") and key in grp:
                    return grp[key][:]
        return None

    with h5py.File(hdf5_path, "r") as f:
        if verbose:
            print(f"📂 File     : {os.path.basename(hdf5_path)}")

        data["metadata"] = dict(f.attrs)

        for band_key, candidates in BAND_CANDIDATES.items():
            result = _find(f, candidates)
            if result is not None:
                data[band_key] = result

    # Konversi band thermal ke float64, mask nilai tidak valid
    for band_key in ["M15", "M16", "I5", "I1", "I2"]:
        if band_key in data:
            data[band_key] = data[band_key].astype(np.float64)
            data[band_key] = np.where(data[band_key] <= 0, np.nan, data[band_key])

    if "lut_i5" in data:
        data["lut_i5"] = data["lut_i5"].astype(np.float64)

    if verbose:
        bands_found = [k for k in ["M15", "M16", "I5"] if k in data]
        print(f"   Band ditemukan : {bands_found if bands_found else 'TIDAK ADA'}")
        for key in ["M15", "M16", "I5", "QC", "QF_I5", "lat", "lon"]:
            if key in data:
                print(f"   {key:<8}: shape={data[key].shape}")
        if "lut_i5" in data:
            lut = data["lut_i5"]
            print(f"   lut_i5  : {len(lut)} entri, BT {lut.min():.1f}–{lut.max():.1f} K")
        if "lat" not in data:
            print("   ℹ️  lat/lon tidak ada di file ini — load via load_geo_i()")

    return data


# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD GEOLOKASI (khusus file VNP03IMG / VJ103IMG)
# ══════════════════════════════════════════════════════════════════════════════

def load_geo_i(geo_path: str, verbose: bool = True) -> dict:
    """
    Load lat, lon dari file geolokasi VNP03IMG / VJ103IMG.

    File geolokasi terpisah dari VNP02IMG — harus download keduanya
    dengan timestamp yang sama dari NASA LAADS DAAC.

    Struktur VNP03IMG:
        /geolocation_data/
            latitude           → float32, per piksel
            longitude          → float32, per piksel

    Parameters
    ----------
    geo_path : str  — path ke file VNP03IMG / VJ103IMG (.h5)
    verbose  : bool — tampilkan info jika True

    Returns
    -------
    dict:
        'lat' : np.ndarray float64 — latitude per piksel
        'lon' : np.ndarray float64 — longitude per piksel
    """
    data = {}

    with h5py.File(geo_path, "r") as f:
        if verbose:
            print(f"📂 Geo file : {os.path.basename(geo_path)}")

        geo = f["geolocation_data"]
        data["lat"] = geo["latitude"][:].astype(np.float64)
        data["lon"] = geo["longitude"][:].astype(np.float64)
        data["solar_zenith"] = geo["solar_zenith_angle"][:].astype(np.float64) \
                               if "solar_zenith_angle" in geo else None

    if verbose:
        print(f"   lat : shape={data['lat'].shape}, "
              f"range [{np.nanmin(data['lat']):.2f}, {np.nanmax(data['lat']):.2f}]")
        print(f"   lon : shape={data['lon'].shape}, "
              f"range [{np.nanmin(data['lon']):.2f}, {np.nanmax(data['lon']):.2f}]")
        if data["solar_zenith"] is not None:
            print(f"   SZA : shape={data['solar_zenith'].shape}")

    return data


def get_scale_offset(hdf5_path: str, band: str = "I05") -> tuple:
    """
    Extract scale_factor dan add_offset dari metadata band di file VNP02IMG.

    Parameters
    ----------
    hdf5_path : str — path ke file VNP02IMG
    band      : str — nama band, default 'I05'

    Returns
    -------
    tuple : (scale_factor, add_offset)
    """
    with h5py.File(hdf5_path, "r") as f:
        ds = f["observation_data"][band]
        scale  = ds.attrs.get("scale_factor", [1.0])
        offset = ds.attrs.get("add_offset",   [0.0])
        scale  = float(scale[0]) if hasattr(scale,  "__len__") else float(scale)
        offset = float(offset[0]) if hasattr(offset, "__len__") else float(offset)
    return scale, offset


# ══════════════════════════════════════════════════════════════════════════════
# 12. SAVE LST GEOTIFF
# ══════════════════════════════════════════════════════════════════════════════

def save_lst_geotiff(lst_celsius: np.ndarray,
                     lat: np.ndarray,
                     lon: np.ndarray,
                     output_path: str,
                     extra_bands: dict = None) -> str:
    """
    Simpan array LST (dan band tambahan opsional) ke GeoTIFF georeferenced.

    Parameters
    ----------
    lst_celsius  : np.ndarray — LST dalam °C (band utama)
    lat          : np.ndarray — latitude per piksel (2D, shape sama dengan LST)
    lon          : np.ndarray — longitude per piksel (2D)
    output_path  : str        — path output file .tif
    extra_bands  : dict       — band tambahan opsional, contoh:
                                {
                                    'delta_t'   : delta_t_array,
                                    'z_score'   : z_score_array,
                                    'lst_sc_celsius': lst_sc_celsius_array,
                                }

    Returns
    -------
    str : path file output

    Catatan: CRS EPSG:4326 (WGS84). Band pertama selalu LST_Celsius.
    """
    lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)
    lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)

    rows, cols = lst_celsius.shape
    transform  = from_bounds(lon_min, lat_min, lon_max, lat_max, cols, rows)

    bands_to_write = {"LST_Celsius": lst_celsius}
    if extra_bands:
        bands_to_write.update(extra_bands)

    profile = {
        "driver"   : "GTiff",
        "dtype"    : rasterio.float32,
        "width"    : cols,
        "height"   : rows,
        "count"    : len(bands_to_write),
        "crs"      : CRS.from_epsg(4326),
        "transform": transform,
        "nodata"   : np.nan,
        "compress" : "lzw",
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        for i, (band_name, band_data) in enumerate(bands_to_write.items(), start=1):
            dst.write(band_data.astype(np.float32), i)
            dst.update_tags(i, name=band_name)

    print(f"💾 Tersimpan : {output_path}")
    print(f"   Bands     : {list(bands_to_write.keys())}")
    print(f"   Extent    : [{lon_min:.4f}, {lat_min:.4f}, {lon_max:.4f}, {lat_max:.4f}]")
    return output_path
def composite_mean_lst(tif_paths: list, output_path: str) -> str:
    """Stack beberapa GeoTIFF LST dan hitung mean per piksel."""
    import rasterio
    import numpy as np
    from rasterio.enums import Resampling

    # Baca semua file, stack jadi array 3D
    arrays = []
    profile = None
    for path in tif_paths:
        with rasterio.open(path) as src:
            arrays.append(src.read(1).astype(np.float32))
            if profile is None:
                profile = src.profile

    stack    = np.stack(arrays, axis=0)           # (n_granule, rows, cols)
    mean_lst = np.nanmean(stack, axis=0)           # mean per piksel, ignore NaN

    profile.update(count=1, dtype=rasterio.float32, nodata=np.nan)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mean_lst, 1)

    print(f"💾 Composite saved: {output_path} ({len(arrays)} granules)")
    return output_path