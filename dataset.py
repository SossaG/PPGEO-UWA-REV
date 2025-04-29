# dataset.py - PyTorch reimplementation of Eric's npy-based grayscale dataset with augmentation

import os
import numpy as np
import torch
from torch.utils.data import Dataset

import cv2

class EGLintonDataset(Dataset):
    # dataset.py - PyTorch reimplementation of Eric's npy-based grayscale dataset with augmentation

    def __init__(self, cfg, subset='train'):
        self.cfg = cfg
        self.subset = subset

        # dataset loading logic
        data_dir = os.path.join(os.path.dirname(__file__), cfg['dataset']['sorted_data_path'])
        self.dataset_idx_list = cfg['dataset']['dataset_idx_list']
        self.dataset_mapping = cfg['dataset']['dataset_mapping']
        self.behavior_lists = cfg['dataset']['behavior_lists']

        self.files = []
        self._populate_files(data_dir)
        self.batch_size = cfg['training']['batch_size']

        # split dataset
        train_ratio = cfg['training']['train_ratio']
        valid_ratio = cfg['training']['valid_ratio']
        test_ratio = 1 - train_ratio - valid_ratio

        total_len = len(self.files)
        train_end = int(train_ratio * total_len)
        val_end = train_end + int(valid_ratio * total_len)

        if subset == 'train':
            self.files = self.files[:train_end]
            # Apply train_coeff if specified
            if 'train_coeff' in cfg['training']:
                coeff = cfg['training']['train_coeff']
                limit = int(len(self.files) * coeff)
                self.files = self.files[:limit]

        elif subset == 'val':
            self.files = self.files[train_end:val_end]
            # Apply valid_coeff if specified
            if 'valid_coeff' in cfg['training']:
                coeff = cfg['training']['valid_coeff']
                limit = int(len(self.files) * coeff)
                self.files = self.files[:limit]

        else:
            self.files = self.files[val_end:]
            # (you could apply test_coeff here too if needed)

        self.aug_cfg = cfg['augmentation']

    def _populate_files(self, base_path):
        for idx in self.dataset_idx_list:
            base_folder = os.path.join(base_path, self.dataset_mapping[idx])

            for cmd_key, cmd_spec in self.behavior_lists.items():
                if cmd_key == 'main':
                    continue
                for behavior in cmd_spec['list']:
                    behavior_path = os.path.join(base_folder, behavior)
                    if not os.path.exists(behavior_path):
                        continue
                    for root, _, files in os.walk(behavior_path):
                        for f in files:
                            if f.endswith('.npy'):
                                self.files.append(os.path.join(root, f))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npy = np.load(self.files[idx], allow_pickle=True)

        # Handle different npy formats
        if len(npy) == 10:
            image, speed, steering = npy[0], npy[8], npy[9]
        elif len(npy) == 8:
            image, speed, steering = npy[0], npy[2], npy[3]
        elif len(npy) == 5:
            image, speed, steering = npy[0], npy[1], npy[2]
        else:
            new_idx = np.random.randint(0, len(self.files))
            return self.__getitem__(new_idx)

        # --- Skip very small images BEFORE resize ---
        if image.shape[0] < 230 or image.shape[1] < 400:
            new_idx = np.random.randint(0, len(self.files))
            return self.__getitem__(new_idx)
        # ---------------------------------------------

        # Center crop if needed
        if image.shape[1] >= 440:
            image = image[:, 40:440]

        # Resize safely
        image = cv2.resize(image, (400, 240))

        # --- After resize, validate final shape ---
        if image.shape != (240, 400):
            new_idx = np.random.randint(0, len(self.files))
            return self.__getitem__(new_idx)
        # -------------------------------------------

        # Normalize
        image = ((image / 127.5) - 1.0).astype(np.float32)
        image = np.expand_dims(image, axis=0)  # (1, 240, 400)

        # Augmentations
        if self.aug_cfg['augment_data'] and np.random.rand() < self.aug_cfg['augment_prob']:
            image = self.apply_augmentations(image)

        return torch.tensor(image), torch.tensor([speed], dtype=torch.float32), torch.tensor([steering], dtype=torch.float32)







    def apply_augmentations(self, image):
        if self.aug_cfg['add_blur']:
            k = np.random.randint(1, self.aug_cfg['blur_range'] + 1)
            if k % 2 == 0: k += 1  # must be odd
            image = cv2.GaussianBlur(image, (k, k), 0)

        if self.aug_cfg['adjust_brightness']:
            factor = np.random.uniform(1 - self.aug_cfg['brightness_range'], 1 + self.aug_cfg['brightness_range'])
            image = np.clip(image * factor, 0, 255).astype(np.uint8)

        if self.aug_cfg['add_noise']:
            noise = np.random.randn(*image.shape) * 255 * self.aug_cfg['noise_range']
            image = np.clip(image + noise, 0, 255).astype(np.uint8)

        if self.aug_cfg['horizontal_shift']:
            max_shift = 80
            i = np.random.randint(0, max_shift)
            shift = int((40 - i) * self.aug_cfg['steering_shift_factor'])
            image = image[:, i:(400+i)]
        return image
