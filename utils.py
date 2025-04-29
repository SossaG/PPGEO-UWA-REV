# utils.py - utility functions for config loading, logging, and callbacks

import yaml
import os
import torch
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_callbacks(cfg, save_dir):
    callbacks = {}

    for cb in cfg['training']['callbacks']:
        if cb['type'] == 'ReduceLROnPlateau':
            callbacks['lr_scheduler_config'] = cb  # Just store the config, not the scheduler yet

        elif cb['type'] == 'EarlyStopping':
            callbacks['early_stopping'] = {
                'monitor': cb['monitor'],
                'patience': cb['patience'],
                'mode': cb.get('mode', 'min'),
                'counter': 0,
                'best': float('inf') if cb.get('mode', 'min') == 'min' else -float('inf')
            }

        elif cb['type'] == 'ModelCheckpoint':
            callbacks['checkpoint_path'] = os.path.join(save_dir, cb['filepath'])

        elif cb['type'] == 'TensorBoard':
            from torch.utils.tensorboard import SummaryWriter
            callbacks['writer'] = SummaryWriter(log_dir=os.path.join(save_dir, cb['log_dir']))

    return callbacks



def log_metrics(epoch, train_loss, val_loss, save_dir):
    log_file = os.path.join(save_dir, "training_log.txt")
    with open(log_file, 'a') as f:
        f.write(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}\n")
    print(f"[Epoch {epoch}] Train: {train_loss:.4f}, Val: {val_loss:.4f}")


