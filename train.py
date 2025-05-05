
# This is a PyTorch reimplementation of Eric's eglinton_train.py logic
# using ResNet-34 as backbone, preserving all training, augmentation, logging, 
# checkpointing, and dataset traversal logic from the original TensorFlow+Hydra pipeline

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
from datetime import datetime
import logging
import wandb

from dataset import EGLintonDataset
from models import ResNet34PilotNet
from utils import load_config, build_callbacks, log_metrics


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_loss_steer = 0.0
    total_loss_speed = 0.0
    for images, speed_labels, steer_labels in dataloader:
        images = images.to(device)
        speed_labels = speed_labels.to(device)
        steer_labels = steer_labels.to(device)

        optimizer.zero_grad()
        pred_speed, pred_steer = model(images)

        loss_speed = criterion(pred_speed, speed_labels)
        loss_steer = criterion(pred_steer, steer_labels)
        loss_weights = cfg['model']['compile'].get('loss_weights', [1.0, 1.0])
        loss = loss_weights[0] * loss_speed + loss_weights[1] * loss_steer
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_loss_steer += loss_steer.item()
        total_loss_speed += loss_speed.item()
    return total_loss / len(dataloader), total_loss_steer / len(dataloader), total_loss_speed / len(dataloader)


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_loss_steer = 0.0
    total_loss_speed = 0.0
    with torch.no_grad():
        for images, speed_labels, steer_labels in dataloader:
            images = images.to(device)
            speed_labels = speed_labels.to(device)
            steer_labels = steer_labels.to(device)

            pred_speed, pred_steer = model(images)

            loss_speed = criterion(pred_speed, speed_labels)
            loss_steer = criterion(pred_steer, steer_labels)
            loss = loss_speed + loss_steer

            total_loss += loss.item()
            total_loss_steer += loss_steer.item()
            total_loss_speed += loss_speed.item()
    return total_loss / len(dataloader), total_loss_steer / len(dataloader), total_loss_speed / len(dataloader)


def custom_collate_fn(batch):
    batch = [sample for sample in batch if sample[0].shape == (1, 240, 400)]

    if len(batch) == 0:
        # No valid samples, return dummy tensors to avoid crashing
        dummy_img = torch.zeros((1, 240, 400), dtype=torch.float32)
        dummy_speed = torch.zeros((1,), dtype=torch.float32)
        dummy_steer = torch.zeros((1,), dtype=torch.float32)
        return dummy_img.unsqueeze(0), dummy_speed.unsqueeze(0), dummy_steer.unsqueeze(0)

    images, speed_labels, steer_labels = zip(*batch)
    images = torch.stack(images, 0)
    speed_labels = torch.stack(speed_labels, 0)
    steer_labels = torch.stack(steer_labels, 0)

    return images, speed_labels, steer_labels


def main():
    cfg = load_config("conf/config.yaml")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
    model_name = f"{cfg['model']['name']}_{timestamp}"
    save_dir = os.path.join(cfg['training']['save_model_dir'], model_name)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet34PilotNet(pretrained=cfg['model']['pretrained']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(cfg['model']['compile']['optimizer']['learning_rate']))
    # Load callbacks (logging, early stopping, LR scheduler, checkpointing)
    callbacks = build_callbacks(cfg, save_dir, optimizer)
    scheduler = callbacks['lr_scheduler']
    criterion = nn.L1Loss()

    # Load datasets
    train_dataset = EGLintonDataset(cfg, subset='train')
    val_dataset = EGLintonDataset(cfg, subset='val')

    print(f"Train dataset size: {len(train_dataset)} samples")
    print(f"Validation dataset size: {len(val_dataset)} samples")

    batch_size = cfg['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=4, pin_memory=True)

    # Initialize Weights & Biases
    wandb.init(
        project=cfg['wandb']['project'],
        name=cfg['wandb']['name'],
        config=cfg,
        dir=save_dir,
        mode=cfg['wandb'].get('mode', 'online')
    )

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(cfg['training']['epochs']):
        train_loss, train_loss_steering, train_loss_speed = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_loss_steering, val_loss_speed = validate_epoch(model, val_loader, criterion, device)

        log_metrics(epoch, train_loss, val_loss, save_dir,
                    train_loss_steering, train_loss_speed,
                    val_loss_steering, val_loss_speed,
                    optimizer.param_groups[0]['lr'])

        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_loss_steering': train_loss_steering,
            'train_loss_speed': train_loss_speed,
            'val_loss': val_loss,
            'val_loss_steering': val_loss_steering,
            'val_loss_speed': val_loss_speed,
            'loss': train_loss + val_loss,
            'lr': optimizer.param_groups[0]['lr']
        }, step=epoch)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, f"{cfg['model']['name']}.pt"))
        else:
            patience_counter += 1

        if patience_counter > cfg['training']['callbacks'][1]['patience']:
            logging.info(f"Early stopping at epoch {epoch}")
            break

    wandb.finish()


if __name__ == '__main__':
    main()
