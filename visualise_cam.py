import torch
import cv2
import numpy as np
from torchvision import transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from models import ResNet34PilotNet

# === Load your trained model ===
ckpt_path = "saved_models_logs/ResNet34PilotNet_2025-05-07-14.38.08/ResNet34PilotNet.pt"  # path on sim pc as this isnt stored in github
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Instantiate model with PPGeo encoder ===
model = ResNet34PilotNet(use_rgb=True).to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

# === Choose the final ResNet conv layer ===
target_layer = model.backbone.layer4[-1]

# === Prepare the CAM object ===
cam = EigenCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())

# === Load and preprocess an EGLinton sample image ===
img_path = "frame.jpg"  # 🔁 Replace with a sample .jpg or extract from a .npy if needed
img = cv2.imread(img_path)[:, :, ::-1]  # BGR to RGB
img = cv2.resize(img, (400, 240))       # Match your input shape
rgb_img = np.float32(img) / 255

# === Apply standard PPGeo preprocessing ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
input_tensor = transform(img).unsqueeze(0).to(device)

# === Generate the attention map ===
grayscale_cam = cam(input_tensor=input_tensor)[0]
cam_output = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# === Save or display the CAM result ===
cv2.imwrite("ppgeo_cam_output.jpg", cam_output)
print("✅ CAM visualisation saved as ppgeo_cam_output.jpg")
