import torch
import cv2
import numpy as np
from torchvision import transforms
from models import ResNet34PilotNet
import yaml
import os

# === CONFIG ===
# For raw PPGeo encoder saliency map:
ckpt_path = "resnet34.ckpt"  # raw encoder checkpoint
img_path = "eglinton images/eglinton image 3 split.jpg"
output_path = "saliency map images/ppgeo raw encoder saliency image 3 split.png"
target_output = "steering"  # Options: "steering" or "speed"

# === Load config ===
with open("conf/config.yaml", 'r') as f:
    cfg = yaml.safe_load(f)

use_rgb = cfg['model'].get('rgb_input', False)

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === OPptional: Load only PPGeo encoder weights into model === comment this section out for non raw encoder
base_model = ResNet34PilotNet(use_rgb=True).to(device)
full_ckpt = torch.load(ckpt_path, map_location=device)
ppgeo_ckpt = full_ckpt["state_dict"] if "state_dict" in full_ckpt else full_ckpt
encoder_weights = {k: v for k, v in ppgeo_ckpt.items() if not k.startswith("fc.")}
base_model.backbone.load_state_dict(encoder_weights, strict=True)
base_model.eval()

# === Optional: comment out below when visualising raw encoder only ===
# base_model = ResNet34PilotNet(use_rgb=use_rgb).to(device)
# state_dict = torch.load(ckpt_path, map_location=device)
# base_model.load_state_dict(state_dict)
# base_model.eval()

# === Wrap model to return selected output ===
class OutputSelector(torch.nn.Module):
    def __init__(self, model, output_type):
        super().__init__()
        self.model = model
        self.output_type = output_type

    def forward(self, x):
        speed, steer = self.model(x)
        return steer if self.output_type == "steering" else speed

model = OutputSelector(base_model, target_output)

# === Load and preprocess image ===
if not os.path.exists(img_path):
    raise FileNotFoundError(f"❌ Image not found at: {img_path}")

if use_rgb:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR → RGB
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

# === Compute saliency map ===
model.zero_grad()
output = model(input_tensor)
output.backward(torch.ones_like(output))
saliency = input_tensor.grad.data.abs().squeeze().cpu().numpy()

if use_rgb:
    print("Saliency raw shape (before mean):", saliency.shape)
    saliency = np.mean(saliency, axis=0)  # average across channels for smoother result
else:
    saliency = saliency[0]  # single channel

# === Normalize saliency for display ===
saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
saliency_colored = cv2.applyColorMap(np.uint8(255 * saliency), cv2.COLORMAP_JET)
saliency_colored = cv2.cvtColor(saliency_colored, cv2.COLOR_BGR2RGB)

# === Resize for alignment ===
original_bgr = cv2.cvtColor((image.squeeze(-1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) if not use_rgb else (image * 255).astype(np.uint8)
saliency_colored = cv2.resize(saliency_colored, (original_bgr.shape[1], original_bgr.shape[0]))

# === Save output ===
os.makedirs(os.path.dirname(output_path), exist_ok=True)
side_by_side = np.concatenate((original_bgr, saliency_colored), axis=1)
cv2.imwrite(output_path, side_by_side)
print(f"✅ Saved saliency map to: {output_path}")
