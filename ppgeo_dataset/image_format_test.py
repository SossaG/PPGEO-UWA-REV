from PIL import Image
import torchvision.transforms as T
import torch

img = Image.open("0.jpg")
transform = T.ToTensor()
tensor = transform(img)  # shape: [3, 160, 320]
print("Shape:", tensor.shape)
print("Pixel value range:", tensor.min().item(), "-", tensor.max().item())

