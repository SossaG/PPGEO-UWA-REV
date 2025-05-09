import torch
import cv2
import numpy as np
from torchvision import transforms
from models import ResNet34PilotNet
import yaml
import os

# === CONFIG ===
ckpt_path = "saved_models_logs/Imagenet pretrained/ResNet34PilotNet.pt"
img_path = "eglinton images/eglinton image 4 diverge.jpg"
output_path = "saliency map images/ trans imagenet saliency image4 diverge.png"

# === Load config ===
with open("conf/config.yaml", 'r') as f:
    cfg = yaml.safe_load(f)

use_rgb = cfg['model'].get('rgb_input', False)

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

# === Compute both saliency maps ===
class OutputSelector(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

model = OutputSelector(base_model)

# === Forward for speed saliency ===
model.zero_grad()
speed_output, _ = model(input_tensor)
speed_output.backward(torch.ones_like(speed_output), retain_graph=True)
speed_saliency = input_tensor.grad.data.abs().squeeze().cpu().numpy()
input_tensor.grad.zero_()

# === Forward for steering saliency ===
input_tensor.requires_grad_()
model.zero_grad()
_, steering_output = model(input_tensor)
steering_output.backward(torch.ones_like(steering_output))
steering_saliency = input_tensor.grad.data.abs().squeeze().cpu().numpy()

# === Combine saliency maps ===
if speed_saliency.ndim == 3:
    speed_saliency = np.mean(speed_saliency, axis=0)
if steering_saliency.ndim == 3:
    steering_saliency = np.mean(steering_saliency, axis=0)

combined_saliency = (speed_saliency + steering_saliency) / 2.0
combined_saliency = (combined_saliency - combined_saliency.min()) / (combined_saliency.max() - combined_saliency.min() + 1e-8)
saliency_colored = cv2.applyColorMap(np.uint8(255 * combined_saliency), cv2.COLORMAP_JET)

# === Base visuals ===
original_bgr = cv2.cvtColor((image.squeeze(-1) * 255).astype(np.uint8),
                            cv2.COLOR_GRAY2BGR) if not use_rgb else (image * 255).astype(np.uint8)
blue_base = original_bgr.copy()  # translucent overlay on real image instead of flat blue  # lighter blue tint for clearer overlay  # blue background in BGR
saliency_overlay = cv2.addWeighted(blue_base, 0.3, saliency_colored, 0.7, 0)  # stronger saliency blend

# === Save result side-by-side ===
os.makedirs(os.path.dirname(output_path), exist_ok=True)
side_by_side = np.concatenate((original_bgr, saliency_overlay), axis=1)
cv2.imwrite(output_path, side_by_side)
print(f"✅ Saved combined speed/steering saliency overlay to: {output_path}")
