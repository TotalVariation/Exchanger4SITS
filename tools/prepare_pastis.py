import os
import numpy as np
import zarr
from tqdm import tqdm
import argparse

def convert_npy_to_zarr(input_dir):
    files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]

    for filename in tqdm(files, desc="Converting .npy to .zarr"):
        npy_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        zarr_path = os.path.join(input_dir, f"{base_name}.zarr")

        array = np.load(npy_path)
        zarr.save(zarr_path, array)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .npy files to .zarr format (in-place).")
    parser.add_argument("input_dir", help="Directory containing .npy files.")

    args = parser.parse_args()
    convert_npy_to_zarr(args.input_dir)

