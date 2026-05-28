import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import numpy as np
import h5py
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import os

def pair_vnp02_vnp03(data_dir: str) -> list[tuple[str, str]]:
    """
    Pairing otomatis VNP02IMG dengan VNP03IMG berdasarkan
    A<date>.<time>.<collection> (komponen 1–3 dari nama file).

    Returns
    -------
    list of (vnp02_path, vnp03_path)
    """
    vnp02_files = glob.glob(os.path.join(data_dir, "VNP02IMG*.h5")) + \
                  glob.glob(os.path.join(data_dir, "VJ102IMG*.h5"))
    vnp03_files = glob.glob(os.path.join(data_dir, "VNP03IMG*.h5")) + \
                  glob.glob(os.path.join(data_dir, "VJ103IMG*.h5"))

    def extract_key(filepath):
        """Ambil A<date>.<time>.<collection> dari nama file."""
        name = os.path.basename(filepath)
        parts = name.split(".")
        return ".".join(parts[1:4])  # A2025015.0130.002

    # Buat lookup dict: key → path
    vnp03_lookup = {extract_key(p): p for p in vnp03_files}

    pairs = []
    unmatched = []
    for vnp02 in sorted(vnp02_files):
        key = extract_key(vnp02)
        if key in vnp03_lookup:
            pairs.append((vnp02, vnp03_lookup[key]))
        else:
            unmatched.append(vnp02)

    print(f"✅ Paired   : {len(pairs)} granule")
    if unmatched:
        print(f"⚠️  No match : {len(unmatched)} file VNP02IMG tanpa pasangan VNP03IMG")
        for f in unmatched:
            print(f"   - {os.path.basename(f)}")

    return pairs


def clip_geotiff(input_path, output_path, bbox):
    """
    Clip GeoTIFF berdasarkan bounding box.

    bbox : tuple — (xmin, ymin, xmax, ymax)
    """
    xmin, ymin, xmax, ymax = bbox
    geom = [box(xmin, ymin, xmax, ymax).__geo_interface__]

    with rasterio.open(input_path) as src:
        out_image, out_transform = mask(src, geom, crop=True)
        out_meta = src.meta.copy()

    out_meta.update({
        "height"   : out_image.shape[1],
        "width"    : out_image.shape[2],
        "transform": out_transform,
    })

    with rasterio.open(output_path, "w", **out_meta) as dst:
        dst.write(out_image)

    print(f"💾 Tersimpan : {output_path}")
    print(f"   Extent    : [{xmin}, {ymin}, {xmax}, {ymax}]")

