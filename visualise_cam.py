import torch
import cv2
import numpy as np
from torchvision import transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from models import ResNet34PilotNet
import yaml

# === CONFIG ===
ckpt_path = "saved_models_logs/Imagenet pretrained/ResNet34PilotNet.pt"  # path on sim pc
img_path = "eglinton images/eglinton image 2.jpg"  # grayscale or RGB input from Eric
output_path = "attention map images/attention map imagenet image 2.png"

# === Load config ===
with open("conf/config.yaml", 'r') as f:
    cfg = yaml.safe_load(f)

use_rgb = cfg['model'].get('rgb_input', False)

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet34PilotNet(use_rgb=use_rgb).to(device)
state_dict = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# === Wrap model to return only steering output ===
class SteeringOnlyModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        _, steering = self.model(x)
        return steering

model = SteeringOnlyModel(model)

# === Target layer for CAM ===
target_layer = model.model.backbone.layer4[-1]
cam = EigenCAM(model=model, target_layers=[target_layer])

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
print("📐 CAM shape:", grayscale_cam.shape, "| type:", type(grayscale_cam))


if use_rgb:
    base_image = image  # shape: [240, 400, 3]
else:
    base_image = np.repeat(image, 3, axis=-1)  # shape: [240, 400, 1] → [240, 400, 3]

cam_image = show_cam_on_image(base_image, grayscale_cam, use_rgb=True)


original_bgr = cv2.cvtColor((image.squeeze(-1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) if not use_rgb else (image * 255).astype(np.uint8)
side_by_side = np.concatenate((original_bgr, cam_image), axis=1)
cv2.imwrite(output_path, side_by_side)
print(f"✅ Saved CAM visualisation to: {output_path}")
