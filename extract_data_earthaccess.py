# ============================================
# extract_earthaccess.py
# ============================================

import earthaccess
import os


def login_earthaccess():
    """
    Login ke NASA Earthdata
    """
    
    earthaccess.login()


def search_viirs(
    short_name="VJ121IMG",
    start_date="2025-01-01",
    end_date="2025-01-10",
    bounding_box=(112.90, -8.10, 113.10, -7.90),
):
    """
    Search dataset VIIRS berdasarkan:
    - dataset name
    - temporal range
    - bbox
    
    Parameters
    ----------
    short_name : str
        Nama dataset VIIRS
        
    start_date : str
        Format YYYY-MM-DD
        
    end_date : str
        Format YYYY-MM-DD
        
    bounding_box : tuple
        (xmin, ymin, xmax, ymax)
        
    Returns
    -------
    results : list
        List granule hasil search
    """

    results = earthaccess.search_data(
        short_name=short_name,
        temporal=(start_date, end_date),
        bounding_box=bounding_box,
    )

    print(f"Jumlah granule ditemukan: {len(results)}")

    return results


def download_viirs(
    results,
    save_dir="/content/drive/MyDrive/Thermal_Processing/data/",
):
    """
    Download granule ke Google Drive
    
    Parameters
    ----------
    results : list
        Hasil search_data
        
    save_dir : str
        Folder penyimpanan
    """

    os.makedirs(save_dir, exist_ok=True)

    earthaccess.download(
        results,
        local_path=save_dir
    )

    print("Download selesai!")

def search_download_data(
    short_name,
    start_date,
    end_date,
    bounding_box,
    save_dir,
):
    """
    Search dan download dataset VIIRS
    
    short_name : str
        Nama dataset
        
    start_date : str
        Format YYYY-MM-DD
        
    end_date : str
        Format YYYY-MM-DD
        
    bounding_box : tuple
        (xmin, ymin, xmax, ymax)
        
    save_dir : str
        Folder penyimpanan
    """

    results = earthaccess.search_data(
        short_name=short_name,
        temporal=(start_date, end_date),
        bounding_box=bounding_box,
    )

    print(f"Jumlah granule ditemukan: {len(results)}")

    os.makedirs(save_dir, exist_ok=True)

    earthaccess.download(
        results,
        local_path=save_dir
    )

    print("Download selesai!")

    return results


    return results