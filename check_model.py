from models import ResNet34PilotNet
from utils import load_config

cfg = load_config("conf/config.yaml")  # Adjust if your path differs
model = ResNet34PilotNet(
    pretrained=cfg['model']['pretrained'],
    use_rgb=cfg['model'].get('use_ppgeo_pretrained_encoder', False)
)

print(model)
