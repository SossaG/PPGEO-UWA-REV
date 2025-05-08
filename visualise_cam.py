import torch
import cv2
import numpy as np
from torchvision import transforms
from pytorch_grad_cam import EigenCAM, GradCAM, GradCAMPlusPlus, ScoreCAM, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from models import ResNet34PilotNet
import yaml

# === CONFIG ===
# === CONFIG ===
ckpt_path = "saved_models_logs/Early Stop Unfrozen ppgeo/ResNet34PilotNet.pt"
img_path = "eglinton images/eglinton image 2.jpg"
output_path = "attention map images/GradCAM steering ppgeo unfrozen image 2.png"

cam_method = "GradCAM"  # Options: "EigenCAM", "GradCAM", "GradCAMPlusPlus", "ScoreCAM", "LayerCAM"
target_output = "steering"  # Options: "steering" or "speed"

# === Load config ===
with open("conf/config.yaml", 'r') as f:
    cfg = yaml.safe_load(f)

use_rgb = cfg['model'].get('rgb_input', False)

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = ResNet34PilotNet(use_rgb=use_rgb).to(device)
state_dict = torch.load(ckpt_path, map_location=device)
base_model.load_state_dict(state_dict)
base_model.eval()

# === Wrap model to return only selected output ===
class OutputSelector(torch.nn.Module):
    def __init__(self, model, output_type):
        super().__init__()
        self.model = model
        self.output_type = output_type

    def forward(self, x):
        speed, steer = self.model(x)
        return steer if self.output_type == "steering" else speed

model = OutputSelector(base_model, output_type=target_output)

# === Select CAM method ===
cam_class = {
    "EigenCAM": EigenCAM,
    "GradCAM": GradCAM,
    "GradCAMPlusPlus": GradCAMPlusPlus,
    "ScoreCAM": ScoreCAM,
    "LayerCAM": LayerCAM
}[cam_method]

target_layer = base_model.backbone.layer4[-1]
cam = cam_class(model=model, target_layers=[target_layer])

# === Load and preprocess image ===
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
grayscale_cam = cam(input_tensor=input_tensor)[0]
print("🧠 CAM values — min:", grayscale_cam.min(), "max:", grayscale_cam.max())

# === Show CAM ===
if use_rgb:
    base_image = image
else:
    base_image = np.repeat(image, 3, axis=-1)

cam_image = show_cam_on_image(base_image, grayscale_cam, use_rgb=True)
original_bgr = cv2.cvtColor((image.squeeze(-1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) if not use_rgb else (image * 255).astype(np.uint8)
side_by_side = np.concatenate((original_bgr, cam_image), axis=1)
cv2.imwrite(output_path, side_by_side)
print(f"✅ Saved CAM visualisation to: {output_path}")
