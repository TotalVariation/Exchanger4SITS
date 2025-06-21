import os
import csv
import pickle
import argparse
import numpy as np
from tqdm import tqdm

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def compute_stats(file_paths, data_dir, key):
    all_data = []

    for rel_path in tqdm(file_paths, desc=f"Loading data {key}"):
        abs_path = os.path.join(data_dir, rel_path)
        data = load_pickle(abs_path)

        arr = data[key]  # shape (T, H, W, C)
        pixels = arr.reshape(-1, arr.shape[-1])  # shape (N, C)
        all_data.append(pixels)

    all_data = np.concatenate(all_data, axis=0)  # shape (total_pixels, C)
    mean = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0)

    return mean, std

def read_csv(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        return [row[0] for row in reader if row]  # assuming single-column CSV

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Directory containing the pickle files')
    parser.add_argument('csv_file', type=str, help='CSV file with relative paths to the pickle files')
    parser.add_argument('output_dir', type=str, help='Output directory to store computed statistics')
    args = parser.parse_args()

    file_paths = read_csv(args.csv_file)
    mean_x10, std_x10 = compute_stats(file_paths, args.data_dir, 'x10')
    mean_x20, std_x20 = compute_stats(file_paths, args.data_dir, 'x20')

    mean = np.hstack((mean_x10, mean_x20))
    std = np.hstack((std_x10, std_x20))

    print('Channel-wise Mean:', mean)
    print('Channel-wise Std:', std)

    stats = np.vstack((mean, std)).T

    with open(os.path.join(args.output_dir, 'stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)

if __name__ == '__main__':
    main()

