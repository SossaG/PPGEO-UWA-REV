import torch
import torch.nn as nn
import torchvision.models as models

class ResNet34PilotNet(nn.Module):
    def __init__(self, pretrained=False, ckpt_path=None, use_rgb=False):
        super(ResNet34PilotNet, self).__init__()
        self.backbone = models.resnet34(pretrained=pretrained)
        in_channels = 3 if use_rgb else 1
        self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Identity()

        self.regressor = nn.Sequential(
            nn.Linear(512, 200),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU()
        )
        self.speed_head = nn.Linear(10, 1)
        self.steering_head = nn.Linear(10, 1)

        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = self.backbone(x)
        x = self.regressor(x)
        speed = self.speed_head(x)
        steering = self.steering_head(x)
        return speed, steering
