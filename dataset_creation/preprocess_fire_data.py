import os
import re
import numpy as np
import rasterio
from glob import glob
import cv2

# Constants
CHANNEL_MEAN = np.array([
    1200.5798078,  1159.92064999, 1143.34170184,
    1039.89754089, 1278.09946045, 1700.76265718,
    1867.6115259,  1954.14941827, 2041.9289693,
    2067.12536653, 1852.88762773, 1452.18596822
], dtype=np.float32)

CHANNEL_STD = np.array([
     719.65335679,  697.77018033,  615.41951954,
     524.34522633,  461.88492677,  392.42986072,
     400.74302951,  414.05815035,  411.44060477,
     388.64697021,  793.58080418, 1397.62723589
], dtype=np.float32)


TARGET_SIZE = (224, 224)
TARGET_DTYPE = np.float32

def convert_tif_to_npz(tif_path: str, out_dir: str, label: int = 1):
    """
    Opens a TIFF file, normalizes its channels, and saves as compressed .npz
    with fields 'image' and 'label'.
    """
    with rasterio.open(tif_path) as src:
        img = src.read().transpose(1, 2, 0).astype(np.float32)  # (H, W, C)

    c = img.shape[-1]
    img = (img - CHANNEL_MEAN[:c]) / CHANNEL_STD[:c]

    filename = os.path.basename(tif_path).replace(".tif", ".npz")
    out_path = os.path.join(out_dir, filename)
    np.savez_compressed(out_path, image=img, label=label)
    print(f"TIFF to NPZ saved: {out_path}")

def convert_all_tifs_in_folder(input_dir: str, output_dir: str, label: int = 1):
    os.makedirs(output_dir, exist_ok=True)
    tif_files = glob(os.path.join(input_dir, "*.tif"))
    if not tif_files:
        print(f"No TIFF files in directory: {input_dir}")
        return
    for tif_path in tif_files:
        try:
            convert_tif_to_npz(tif_path, output_dir, label)
        except Exception as e:
            print(f"Error converting {tif_path}: {e}")

def process_npz_file(path: str, output_dir: str):
    """
    Loads a .npz file, removes 'aerosol' array if present, casts to float32,
    resizes to TARGET_SIZE, and re-saves as compressed .npz.
    """
    data = np.load(path)
    image = data['image']
    label = int(data['label'])

    image = image.astype(TARGET_DTYPE)

    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    image = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    filename = os.path.basename(path)
    out_path = os.path.join(output_dir, filename)
    np.savez_compressed(out_path, image=image, label=label)
    print(f"Processed NPZ saved: {out_path}")

def batch_process_npz_folder(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    npz_files = glob(os.path.join(input_dir, "*.npz"))
    if not npz_files:
        print(f"No NPZ files in directory: {input_dir}")
        return
    for npz_path in npz_files:
        try:
            process_npz_file(npz_path, output_dir)
        except Exception as e:
            print(f"Error processing {npz_path}: {e}")

def rename_npz_files(input_dir: str):
    """
    Renames files from:
      ptXXXXX_YYYY-MM-DDThhmm_lat_lon.npz
    to:
      fire_BRA_YYYYMMDDThhmm_lat_lon.npz
    """
    for filename in os.listdir(input_dir):
        if not filename.endswith(".npz"):
            continue
        match = re.match(r'pt\d+_(\d{4}-\d{2}-\d{2}T\d+)_([-\d.]+)_([-\d.]+)\.npz', filename)
        if not match:
            continue
        date_str = match.group(1).replace("-", "")
        lat, lon = match.group(2), match.group(3)
        new_name = f"fire_BRA_{date_str}_{lat}_{lon}.npz"
        src = os.path.join(input_dir, filename)
        dst = os.path.join(input_dir, new_name)
        os.rename(src, dst)
        print(f"Renamed: {filename} → {new_name}")

if __name__ == "__main__":
    RAW_TIF_DIR       = "GCS/BRA_2021_V2/fire_tif"
    INITIAL_NPZ_DIR   = "GCS/BRA_2021_V2/fire_converted"
    PROCESSED_NPZ_DIR = "GCS/BRA_2021_V2/converted_final"

    convert_all_tifs_in_folder(RAW_TIF_DIR, INITIAL_NPZ_DIR, label=1)

    batch_process_npz_folder(INITIAL_NPZ_DIR, PROCESSED_NPZ_DIR)

    rename_npz_files(PROCESSED_NPZ_DIR)
