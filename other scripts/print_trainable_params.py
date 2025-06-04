import torch
from models import ResNet34PilotNet
from utils import load_config

# Load config
cfg = load_config("conf/config.yaml")
use_ppgeo = cfg['model'].get('use_ppgeo_pretrained_encoder', False)
freeze_encoder = cfg['model'].get('freeze_encoder', False)

# Create model instance (no freeze_encoder arg!)
model = ResNet34PilotNet(
    pretrained=cfg['model']['pretrained'],
    use_rgb=use_ppgeo
)

# Freeze encoder manually if requested
if freeze_encoder:
    print("🔒 Freezing encoder weights for parameter count check")
    for param in model.backbone.parameters():
        param.requires_grad = False

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
non_trainable_params = total_params - trainable_params

# Display
print(f"🧠 Total parameters:      {total_params:,}")
print(f"✅ Trainable parameters:  {trainable_params:,}")
print(f"🚫 Non-trainable (frozen): {non_trainable_params:,}")
