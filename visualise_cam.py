import torch
import cv2
import numpy as np
from torchvision import transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from models import ResNet34PilotNet

# === CONFIG ===
ckpt_path = "saved_models_logs/ResNet34PilotNet_2025-05-05-19.55.23/ResNet34PilotNet.pt"  # path on sim pc as this isnt stored in github
img_path = "eglinton images/elington image 1.png"  # ← Path to your grayscale cropped image from Eric
output_path = "attention map images/attention map image 1.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Model ===
model = ResNet34PilotNet(use_rgb=True).to(device)
state_dict = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# === Grad-CAM Setup ===
target_layer = model.backbone.layer4[-1]
cam = EigenCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())

# === Load and Process Image ===
gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
gray = cv2.resize(gray, (400, 240))
rgb_img = np.stack([gray] * 3, axis=-1).astype(np.float32) / 255.0

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
input_tensor = transform(rgb_img).unsqueeze(0).to(device)

# === Generate CAM Overlay ===
grayscale_cam = cam(input_tensor=input_tensor)[0]
cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# === Stack Original and CAM for Side-by-Side Comparison ===
original_bgr = cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
side_by_side = np.concatenate((original_bgr, cam_image), axis=1)

cv2.imwrite(output_path, side_by_side)
print(f"✅ Saved CAM visualisation: {output_path}")

