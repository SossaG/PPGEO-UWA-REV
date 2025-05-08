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
    scheduler_cfg = next(cb for cb in cfg['training']['callbacks'] if cb['type'] == 'ReduceLROnPlateau')

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=scheduler_cfg.get('mode', 'min'),
        factor=float(scheduler_cfg.get('factor', 0.5)),
        patience=int(scheduler_cfg.get('patience', 1)),
        min_lr=float(scheduler_cfg.get('min_lr', 1e-7)), 
        verbose=int(scheduler_cfg.get('verbose', 1))
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
