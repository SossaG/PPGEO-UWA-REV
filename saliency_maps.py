import torch
import cv2
import numpy as np
from torchvision import transforms
from models import ResNet34PilotNet
import yaml
import os

# === CONFIG ===
# For raw PPGeo encoder saliency map:
ckpt_path = "saved_models_logs/Imagenet pretrained/ResNet34PilotNet.pt"  # raw encoder checkpoint
img_path = "eglinton images/eglinton image 2 pull in.jpg"
output_path = "saliency map images/ both imagenet saliency image 2 pull in.png"

# === Load config ===
with open("conf/config.yaml", 'r') as f:
    cfg = yaml.safe_load(f)

use_rgb = cfg['model'].get('rgb_input', False)

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load only PPGeo encoder weights into model ===
"""
base_model = ResNet34PilotNet(use_rgb=True).to(device)
full_ckpt = torch.load(ckpt_path, map_location=device)
ppgeo_ckpt = full_ckpt["state_dict"] if "state_dict" in full_ckpt else full_ckpt
encoder_weights = {k: v for k, v in ppgeo_ckpt.items() if not k.startswith("fc.")}
base_model.backbone.load_state_dict(encoder_weights, strict=True)
base_model.eval()
"""

# === Load full model with config-based RGB setting ===
base_model = ResNet34PilotNet(use_rgb=use_rgb).to(device)
state_dict = torch.load(ckpt_path, map_location=device)
base_model.load_state_dict(state_dict)
base_model.eval()

# === Load and preprocess image ===
if not os.path.exists(img_path):
    raise FileNotFoundError(f"❌ Image not found at: {img_path}")

if use_rgb:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:, :, ::-1]
    img = cv2.resize(img, (400, 240))
    image = img.astype(np.float32) / 255.0
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
else:
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, (400, 240))
    image = np.expand_dims(gray.astype(np.float32) / 255.0, axis=-1)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

input_tensor = transform(image).unsqueeze(0).to(device)
input_tensor.requires_grad_()

# === Compute saliency for steering ===
base_model.zero_grad()
_, steering_output = base_model(input_tensor)
steering_output.backward(torch.ones_like(steering_output), retain_graph=True)
steering_saliency = input_tensor.grad.data.abs().squeeze().cpu().numpy()
input_tensor.grad.zero_()

# === Compute saliency for speed ===
input_tensor.requires_grad_()
base_model.zero_grad()
speed_output, _ = base_model(input_tensor)
speed_output.backward(torch.ones_like(speed_output))
speed_saliency = input_tensor.grad.data.abs().squeeze().cpu().numpy()

# === Process saliency maps (now with eric colour style) ===
def process_saliency(raw):
    if raw.ndim == 3:
        raw = np.mean(raw, axis=0)
    raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
    saliency_colored = cv2.applyColorMap(np.uint8(255 * raw), cv2.COLORMAP_JET)
    blue_base = np.full_like(saliency_colored, (255, 0, 0))  # BGR blue
    return cv2.addWeighted(blue_base, 0.5, saliency_colored, 0.5, 0)


saliency_steering = process_saliency(steering_saliency)
saliency_speed = process_saliency(speed_saliency)

# === Resize for alignment ===
original_bgr = cv2.cvtColor((image.squeeze(-1) * 255).astype(np.uint8),
                            cv2.COLOR_GRAY2BGR) if not use_rgb else (image * 255).astype(np.uint8)
saliency_steering = cv2.resize(saliency_steering, (original_bgr.shape[1], original_bgr.shape[0]))
saliency_speed = cv2.resize(saliency_speed, (original_bgr.shape[1], original_bgr.shape[0]))

# === Save output ===
os.makedirs(os.path.dirname(output_path), exist_ok=True)
combined = np.concatenate((saliency_speed, original_bgr, saliency_steering), axis=1)
cv2.imwrite(output_path, combined)
print(f"✅ Saved combined speed/steering saliency map to: {output_path}")
