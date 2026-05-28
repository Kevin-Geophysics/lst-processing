
def run_pipeline(
        data_path,
        output_path,
        image_path,
        composite_methode = "median"
):
    lst_results = []
    spatial_meta = None  # Tambahan untuk menyimpan metadata spasial satelit

    nc_files = sorted(data_path.glob("*.nc"))

    for file in nc_files:
        print(f"\nProcessing: {file.name}")

        # ==============================================================
        # 1. LOAD DATA & AMBIL METADATA (Misal via fungsi load_viirs_bands kamu)
        # ==============================================================
        bands = load_viirs_bands(file)
        
        # Simpan metadata spasial/geotransform dari file pertama untuk referensi komposit
        if spatial_meta is None and "meta" in bands:
            spatial_meta = bands["meta"]

        # ... (Langkah 2 sampai 6 tetap sama) ...

        # ==============================================================
        # 7. SAVE DAILY GEOTIFF (Pastikan fungsi ini menerima metadata/transform)
        # ==============================================================
        output_file = output_path / f"{file.stem}_lst.tif"
        
        # Pastikan save_geotiff menerima metadata agar koordinat harian aman
        save_geotiff(
            output_file, 
            anomaly["lst_celsius"], 
            meta=bands.get("meta") # Pastikan modul io.py mendukung ini
        )

        lst_results.append(anomaly["lst_celsius"])
        # ...

    # ==================================================================
    # 10. TEMPORAL COMPOSITING
    # ==================================================================
    print("\nCreating temporal composite...")
    lst_stack = np.stack(lst_results)
    lst_composite = composite(lst_stack, method=composite_method)

    # ==================================================================
    # 11. SAVE COMPOSITE (Gunakan spatial_meta agar koordinatnya presisi!)
    # ==================================================================
    composite_file = output_path / "lst_composite.tif"
    save_geotiff(
        composite_file, 
        lst_composite, 
        meta=spatial_meta # Koordinat komposit merujuk ke data asli
    )

    # ==================================================================
    # 11.5 SPATIAL CLIPPING (Tahap baru sebelum Visualisasi)
    # ==================================================================
    print("Clipping composite to AOI...")
    # Contoh memanggil fungsi clip dari modul io atau utils jika ada
    # ao_geometry = "path_to_your_shapefile.shp"
    # lst_clipped, clipped_meta = clip_raster(composite_file, ao_geometry)
    
    # Update file tif yang sudah rapi tercrop
    # clipped_file = output_path / "lst_composite_clipped.tif"
    # save_geotiff(clipped_file, lst_clipped, meta=clipped_meta)

    # ==================================================================
    # 12. SAVE FINAL IMAGE (Gunakan data yang sudah bersih/diclip)
    # ==================================================================
    image_file = image_path / "lst_composite.png"
    plot_lst(
        lst_composite, # ganti dengan lst_clipped jika proses clipping sudah aktif
        save_path=image_file,
        title="LST Composite Agustus"
    )

    print("\nPipeline selesai!")
    return lst_composite
