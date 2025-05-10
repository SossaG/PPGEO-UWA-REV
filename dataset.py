
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

class EGLintonDataset(Dataset):
    def __init__(self, cfg, subset='train'):
        self.cfg = cfg
        self.subset = subset

        data_dir = os.path.join(os.path.dirname(__file__), cfg['dataset']['sorted_data_path'])
        self.dataset_idx_list = cfg['dataset']['dataset_idx_list']
        self.dataset_mapping = cfg['dataset']['dataset_mapping']
        self.behavior_lists = cfg['dataset']['behavior_lists']

        self.files = []
        self._populate_files(data_dir)

        train_ratio = cfg['training']['train_ratio']
        valid_ratio = cfg['training']['valid_ratio']
        test_ratio = 1 - train_ratio - valid_ratio

        # Shuffle the dataset file list before slicing (matches Eric's original logic)
        if self.cfg.get('dataset', {}).get('shuffle', False):
            import random  # Added for shuffling support
            random.shuffle(self.files)  # <-- This enables file-level shuffling before split

        total_len = len(self.files)

        train_end = int(self.cfg['training']['train_ratio'] * total_len)
        val_end = train_end + int(self.cfg['training']['valid_ratio'] * total_len)
        
        if subset == 'train':
            self.files = self.files[:train_end]
            if 'train_coeff' in cfg['training']:
                coeff = cfg['training']['train_coeff']
                self.files = self.files[:int(len(self.files) * coeff)]
        elif subset == 'val':
            self.files = self.files[train_end:val_end]
            if 'valid_coeff' in cfg['training']:
                coeff = cfg['training']['valid_coeff']
                self.files = self.files[:int(len(self.files) * coeff)]
        else:
            self.files = self.files[val_end:]

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

        if len(npy) == 10:
            image, speed, steering = npy[0], npy[8], npy[9]
        elif len(npy) == 8:
            image, speed, steering = npy[0], npy[2], npy[3]
        elif len(npy) == 5:
            image, speed, steering = npy[0], npy[1], npy[2]
        else:
            return self.__getitem__(np.random.randint(0, len(self.files)))

        # Augmentation and preprocessing
        if self.aug_cfg['augment_data']:
            if np.random.rand() < self.aug_cfg['augment_prob']:
                image, steering = self.apply_augmentations(image, steering)

            if self.aug_cfg['horizontal_shift'] or self.aug_cfg['horizontal_rotate']:
                if self.aug_cfg['horizontal_rotate']:
                    image, steering = self.horizontal_rotate(image, steering, self.aug_cfg['steering_rotate_factor'])
                else:
                    image, steering = self.horizontal_shift(image, steering, self.aug_cfg['steering_shift_factor'])
            else:
                image = image[:, 40:440]
        else:
            image = image[:, 40:440]

        image = ((image / 127.5) - 1.0).astype(np.float32)

        # Convert grayscale to RGB only if specified by config
        if self.cfg['model'].get('rgb_input', False):
            if image.ndim == 2:
                image = np.repeat(np.expand_dims(image, axis=0), 3, axis=0)
        else:
            if image.ndim == 2:
                image = np.expand_dims(image, axis=0)
        

        return torch.tensor(image), torch.tensor([speed], dtype=torch.float32), torch.tensor([steering], dtype=torch.float32)

    def apply_augmentations(self, image, steering):
        if self.aug_cfg['horizontal_flip'] and np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            steering = -steering

        if self.aug_cfg['add_blur']:
            k = np.random.randint(1, self.aug_cfg['blur_range'] + 1)
            if k % 2 == 0: k += 1
            image = cv2.GaussianBlur(image, (k, k), 0)

        if self.aug_cfg['adjust_brightness']:
            factor = np.random.uniform(1 - self.aug_cfg['brightness_range'], 1 + self.aug_cfg['brightness_range'])
            image = np.clip(image * factor, 0, 255).astype(np.uint8)

        if self.aug_cfg['add_noise']:
            noise = np.random.randn(*image.shape) * 255 * self.aug_cfg['noise_range']
            image = np.clip(image + noise, 0, 255).astype(np.uint8)

        if self.aug_cfg['add_shadow']:
            h, w = image.shape
            x1, x2 = np.random.randint(0, w, 2)
            shadow_mask = np.zeros_like(image)
            cv2.fillPoly(shadow_mask, np.array([[[x1, 0], [x2, h], [w, h], [0, h]]]), 255)
            shadow_factor = np.random.uniform(1 - self.aug_cfg['shadow_range'], 1)
            image = np.where(shadow_mask == 255, image * shadow_factor, image).astype(np.uint8)

        return image, steering

    def horizontal_shift(self, image, steering, steering_shift_factor):
        i = np.random.randint(0, 80)
        shift_pixels = (40 - i)
        steering -= shift_pixels / 40 * steering_shift_factor
        image = image[:, i:(400+i)]
        return image, steering

    def horizontal_rotate(self, image, steering, steering_rotate_factor):
        i = np.random.randint(0, 80)
        reduc_left = int(20 * (i - 40) / 40) if i > 40 else 0
        reduc_right = int(20 * (40 - i) / 40) if i < 40 else 0

        h, w = image.shape
        pt_A = np.array([0, reduc_left])
        pt_B = np.array([0, h-1 - reduc_left])
        pt_C = np.array([w-1, h-1 - reduc_right])
        pt_D = np.array([w-1, reduc_right])

        input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
        output_pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]])
        M = cv2.getPerspectiveTransform(input_pts, output_pts)
        rotated_image = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR)

        rotated_steering = steering - (40 - i) / 40 * steering_rotate_factor
        return rotated_image, rotated_steering
