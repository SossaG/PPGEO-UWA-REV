# utils.py - utility functions for config loading, logging, and callbacks

import yaml
import os
import torch
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_callbacks(cfg, save_dir, optimizer):
    scheduler_cfg = cfg['callbacks'][0]  # assumes ReduceLROnPlateau is first

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=scheduler_cfg.get('factor', 0.5),
        patience=scheduler_cfg.get('patience', 1),
        min_lr=scheduler_cfg.get('min_lr', 1e-7),
        verbose=scheduler_cfg.get('verbose', 1)
    )

    return {'lr_scheduler': scheduler}


def log_metrics(epoch, train_loss, val_loss, save_dir,
                train_loss_steering, train_loss_speed,
                val_loss_steering, val_loss_speed,
                lr):
    log_str = (
        f"Epoch {epoch}:\n"
        f"  Train Loss: {train_loss:.4f} (Steering: {train_loss_steering:.4f}, speed: {train_loss_speed:.4f})\n"
        f"  Val   Loss: {val_loss:.4f} (Steering: {val_loss_steering:.4f}, speed: {val_loss_speed:.4f})\n"
        f"  Learning Rate: {lr:.8f}\n"
    )
    print(log_str)

    log_path = os.path.join(save_dir, "training_log.txt")
    with open(log_path, "a") as f:
        f.write(log_str + "\n")
