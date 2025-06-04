
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
from models_ivan import ResNet34PilotNet

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


# === Training loop ===
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

# === Validation loop ===
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

# === Custom collate function for error handling ===
def make_collate_fn(cfg):
    def custom_collate_fn(batch):
        expected_channels = 3 if cfg['model'].get('rgb_input', False) else 1
        batch = [sample for sample in batch if sample is not None and sample[0].shape == (expected_channels, 180, 400)]


        if len(batch) == 0:
            # No valid samples, return dummy tensors to avoid crashing
            #print("[WARNING] All samples in this batch are invalid. Returning dummy batch.")
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

# === Model builder pulled out for reuse across cmd_key loop ===
def build_model(cfg, device):
    use_ppgeo = cfg['model'].get('use_ppgeo_pretrained_encoder', False)
    if use_ppgeo:
        print("🟢 Using PPGeo pretrained ResNet-34 encoder")
        ppgeo_ckpt = torch.load('resnet34.ckpt', map_location='cpu')
        state_dict = ppgeo_ckpt['state_dict']
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
        model = ResNet34PilotNet(use_rgb=cfg['model'].get('rgb_input', False)).to(device)

        if cfg['model'].get('freeze_encoder', False):
            print('🔒 Freezing encoder weights')
            for param in model.backbone.parameters():
                param.requires_grad = False

        if cfg['model'].get('partial_freeze', False):
        # freeze all but conv1 and layer1
            print('Partially Freezing encoder weights')
            for name, p in model.backbone.named_parameters():
                if not (name.startswith('conv1') or name.startswith('layer1')):
                    p.requires_grad = False

        # ——— GRAYSCALE ADAPTATION & WEIGHT LOADING ———
        # if doing true-grayscale fine-tuning, average the pretrained RGB conv1 → 1-channel
        if not cfg['model'].get('rgb_input', True):
            # checkpoint has "conv1.weight": torch.Size([64,3,7,7])
            w3 = state_dict['conv1.weight']                # [64,3,7,7]
            state_dict['conv1.weight'] = w3.mean(1, keepdim=True)  # → [64,1,7,7] 
        # now load everything (conv1 will match or be ignored)  
        model.backbone.load_state_dict(state_dict, strict=False)

    else:
        print("🟡 Training from scratch/Imagenet(no PPGeo)")
        model = ResNet34PilotNet(pretrained=cfg['model']['pretrained'], use_rgb=cfg['model'].get('rgb_input', False)).to(device)

        if cfg['model'].get('partial_freeze', False):
        # freeze all but conv1 and layer1
            for name, p in model.backbone.named_parameters():
                if not (name.startswith('conv1') or name.startswith('layer1')):
                    p.requires_grad = False

        # ——— GRAYSCALE ADAPTER for ImageNet ———
        if not cfg['model'].get('rgb_input', False):
            # model.backbone.conv1.weight is [64,3,7,7] → average to [64,1,7,7]
            w3 = model.backbone.conv1.weight.data         # [64,3,7,7]
            w1 = w3.mean(1, keepdim=True)                # [64,1,7,7]
            model.backbone.conv1.weight.data.copy_(w1)

    return model

# === Main training launcher ===
def main():
    cfg = load_config("conf/config.yaml")
    cmd_keys = cfg['training'].get('cmd_list', ['cmd_0'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for cmd_key in cmd_keys:
        print(f"\n🔁 Training model for {cmd_key}")

        model_name = f"{cfg['wandb']['name']}_{cmd_key}"
        save_dir = os.path.join(cfg['training']['save_model_dir'], model_name)
        os.makedirs(save_dir, exist_ok=True)

        model = build_model(cfg, device)

        optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(cfg['model']['compile']['optimizer']['learning_rate'])
        )

        callbacks = build_callbacks(cfg, save_dir, optimizer)
        scheduler = callbacks['lr_scheduler']
        criterion = nn.L1Loss()

        start_epoch = 0
        best_val_loss = float('inf')

        if cfg['training'].get('load_checkpoint', False):
            checkpoint_path = os.path.join(save_dir, f'{run_name}_checkpoint.pt')
            if os.path.exists(checkpoint_path):
                print(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                required_keys = ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict']
                missing_keys = [k for k in required_keys if k not in checkpoint]
                if missing_keys:
                    print(f"⚠️ Warning: Missing keys in checkpoint: {missing_keys}")

                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                start_epoch = checkpoint.get('epoch', 0)
                wandb.log({"resumed_from_checkpoint": True}, step=start_epoch)
            else:
                print(f"No checkpoint found at {checkpoint_path}, starting from scratch.")


        # === Load per-cmd dataset ===
        train_dataset = EGLintonDataset(cfg, subset='train', cmd_key=cmd_key)
        val_dataset = EGLintonDataset(cfg, subset='val', cmd_key=cmd_key)

        print(f"Train dataset size: {len(train_dataset)} samples")
        print(f"Validation dataset size: {len(val_dataset)} samples")

        batch_size = cfg['training']['batch_size']
        collate_fn = make_collate_fn(cfg)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

        # === Init Weights & Biases logging ===
        wandb.init(
            project=cfg['wandb']['project'],
            name=f"{cfg['wandb']['name']}_{cmd_key}",
            config=cfg,
            dir=save_dir,
            mode=cfg['wandb'].get('mode', 'online')
        )
        run_name = wandb.run.name  # To be able to use it for saved model name


        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(start_epoch, cfg['training']['epochs']):
            train_loss, train_loss_steering, train_loss_speed = train_epoch(model, train_loader, optimizer, criterion, device, cfg)
            val_loss, val_loss_steering, val_loss_speed = validate_epoch(model, val_loader, criterion, device, cfg)

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
                # checkpoint.pt saves not just weights, but all the other info as well
                checkpoint_path = os.path.join(save_dir, f'{run_name}_checkpoint.pt') 
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss
                }, checkpoint_path)

            else:
                patience_counter += 1

            if patience_counter > cfg['training']['callbacks'][1]['patience']:
                logging.info(f"Early stopping at epoch {epoch}")
                break

        wandb.finish()

if __name__ == '__main__':
    main()
