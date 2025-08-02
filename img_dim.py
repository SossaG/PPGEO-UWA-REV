import os
import numpy as np

# 🔧 CONFIGURATION
dataset_dir = "/media/sim/data/eglinton_datasorting_dual/sorted_eglinton_data/CIL_Dual_Cam_Stage2_First_Half/lane_empty_bay/rosbag2_2024_07_19-11_02_11_0_20117-20264/Img_20124_.npy"  # <<< change this path as needed


data = np.load(dataset_dir, allow_pickle=True)
img = data[0]  # First element should be the front camera image
print(f"{dataset_dir}: shape={img.shape}")
# Optional: break after first example if you just want a quick check
# break
