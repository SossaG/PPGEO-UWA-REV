
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


def train_epoch(model, dataloader, optimizer, criterion, device, cfg):
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


def validate_epoch(model, dataloader, criterion, device, cfg):
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
            loss_weights = cfg['model']['compile'].get('loss_weights', [1.0, 1.0])
            loss = loss_weights[0] * loss_speed + loss_weights[1] * loss_steer


            total_loss += loss.item()
            total_loss_steer += loss_steer.item()
            total_loss_speed += loss_speed.item()
    return total_loss / len(dataloader), total_loss_steer / len(dataloader), total_loss_speed / len(dataloader)

def make_collate_fn(cfg):
    def custom_collate_fn(batch):

        expected_channels = 3 if cfg['model'].get('rgb_input', False) else 1
        batch = [sample for sample in batch if sample[0].shape == (expected_channels, 240, 400)]



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
    return custom_collate_fn

def main():
    cfg = load_config("conf/config.yaml")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
    model_name = f"{cfg['model']['name']}_{timestamp}"
    save_dir = os.path.join(cfg['training']['save_model_dir'], model_name)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    # === Conditional model loading ===
    use_ppgeo = cfg['model'].get('use_ppgeo_pretrained_encoder', False)

    if use_ppgeo:
        print("🟢 Using PPGeo pretrained ResNet-34 encoder")
        ppgeo_ckpt = torch.load('resnet34.ckpt', map_location='cpu')
        state_dict = ppgeo_ckpt['state_dict']
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
        use_rgb = True
        model = ResNet34PilotNet(use_rgb= cfg['model'].get('rgb_input', False)).to(device)
        if cfg['model'].get('freeze_encoder', False):
            print('🔒 Freezing encoder weights')
            for param in model.backbone.parameters():
                param.requires_grad = False
        model.backbone.load_state_dict(state_dict, strict=True)

                # === PPGeo Diagnostic Check ===
        print("🔍 Checking PPGeo encoder state:")
        print("→ conv1.weight shape:", model.backbone.conv1.weight.shape)
        print("→ conv1.weight mean:", model.backbone.conv1.weight.mean().item())

        # Check first few loaded keys
        print("→ First few keys in loaded state_dict:")
        print(list(state_dict.keys())[:5])

        # Confirm a known weight slice (sanity check)
        conv1_sample = model.backbone.conv1.weight[0, 0, 0, :5]
        print("→ conv1[0,0,0,:5]:", conv1_sample.tolist())
        
        # === Quick test to verify encoder is functional ===
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3 if cfg['model'].get('rgb_input', False) else 1, 240, 400).to(device)  # Dummy RGB or grayscale input
            try:
                pred_speed, pred_steer = model(dummy_input)
                print("✅ Forward pass successful.")
                print("   ➤ Speed output shape:", pred_speed.shape)
                print("   ➤ Steer output shape:", pred_steer.shape)
            except Exception as e:
                print("❌ Error during forward pass:", e)
    else:
        print("🟡 Training from scratch/Imagenet(no PPGeo)")
        model = ResNet34PilotNet(pretrained=cfg['model']['pretrained'], use_rgb= cfg['model'].get('rgb_input', False)).to(device)
        print("ImageNet conv1 mean:", model.backbone.conv1.weight.mean().item())


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
    collate_fn = make_collate_fn(cfg)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

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
        train_loss, train_loss_steering, train_loss_speed = train_epoch(model, train_loader, optimizer, criterion, device,cfg)
        val_loss, val_loss_steering, val_loss_speed = validate_epoch(model, val_loader, criterion, device,cfg)

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