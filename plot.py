import numpy as np
import h5py
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import os
import matplotlib.pyplot as plt
def plot_lst(tif_path, band=1, title="LST (°C)", vmin=None, vmax=None):
    """
    Visualisasi LST di atas basemap OpenStreetMap.

    band : int — 1=LST_Celsius, 2=delta_t, 3=z_score
    """
    with rasterio.open(tif_path) as src:
        data      = src.read(band).astype(np.float32)
        data      = np.where(data == src.nodata, np.nan, data)
        extent    = plotting_extent(src)
        crs       = src.crs

    fig, ax = plt.subplots(figsize=(10, 8))

    img = ax.imshow(
        data,
        extent  = extent,
        cmap    = "RdYlBu_r",
        vmin    = vmin or np.nanpercentile(data, 2),
        vmax    = vmax or np.nanpercentile(data, 98),
        alpha   = 0.75,
        zorder  = 2,
    )

    ctx.add_basemap(ax, crs=crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik, zorder=1)

    plt.colorbar(img, ax=ax, label=title, shrink=0.7)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.show()