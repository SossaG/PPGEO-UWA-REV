import os
import numpy as np
import random

def check_dataset(base_path, sample_fraction=0.05):  # check only 5% of the files
    all_npy_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.npy'):
                all_npy_files.append(os.path.join(root, file))

    total_files = len(all_npy_files)
    sample_size = max(1, int(total_files * sample_fraction))
    sample_size = min(sample_size, total_files)  # clamp sample size to available files
    sample_files = random.sample(all_npy_files, sample_size)


    print(f"Found {total_files} .npy files. Checking a random sample of {sample_size} files ({sample_fraction*100:.1f}% of dataset)...")

    bad_files = []

    for file_path in sample_files:
        try:
            npy = np.load(file_path, allow_pickle=True)

            if len(npy) == 10:
                image = npy[0]
            elif len(npy) == 8:
                image = npy[0]
            elif len(npy) == 5:
                image = npy[0]
            else:
                bad_files.append((file_path, 'unexpected npy length'))
                continue

            # Check dimensions
            if image.shape[0] != 240 or image.shape[1] < 400:
                bad_files.append((file_path, f"bad shape {image.shape}"))

        except Exception as e:
            bad_files.append((file_path, f"load error: {e}"))

    print(f"\nChecked {sample_size} files.")
    print(f"Corrupted or bad files found: {len(bad_files)}")

    if bad_files:
        print("\nSample bad files:")
        for path, reason in bad_files[:10]:  # show first 10
            print(f" - {path}: {reason}")

    if len(bad_files) == sample_size:
        print("\n🚨 WARNING: All sampled files are corrupted!")
    elif len(bad_files) > sample_size * 0.5:
        print("\n⚠ WARNING: More than 50% of sampled files have issues!")
    else:
        print("\n✅ Sample looks good. You can proceed to training.")

if __name__ == "__main__":
    # Replace with your dataset path
    check_dataset("/media/sim/data/eglinton_datasorting_dual/sorted_eglinton_data", sample_fraction=0.001)
