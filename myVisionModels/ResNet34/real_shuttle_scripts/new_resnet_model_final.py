# eglinton_nav_model.py
import torch
import torch.nn as nn
from typing import List
# Import the exact encoder you provided (unchanged)
from resnet_encoder import ResnetEncoder  # ResnetEncoder(34, True, num_input_images=1)
class NavHead(nn.Module):
    """
    Lightweight dual-regression head for shuttle navigation.
    Mirrors your previous structure:
      - GAP over layer4 feature map (512 ch)
      - MLP 512 -> hidden -> 64
      - Two 1D heads: speed and steer
    """
    def __init__(self, in_channels: int = 512, hidden: int = 256, p_drop: float = 0.2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 64),
            nn.ReLU(inplace=True),
        )
        self.speed = nn.Linear(64, 1)
        self.steer = nn.Linear(64, 1)
        # small, neutral init for stable starts
        nn.init.zeros_(self.speed.weight); nn.init.zeros_(self.speed.bias)
        nn.init.zeros_(self.steer.weight); nn.init.zeros_(self.steer.bias)
    def forward(self, feats: List[torch.Tensor]):
        x = self.pool(feats[-1]).flatten(1)  # use layer4 (512 ch)
        x = self.mlp(x)
        return self.speed(x), self.steer(x)

class EglintonNavModel(nn.Module):
    """
    Encoder: EXACT ResnetEncoder(34, pretrained=True, num_input_images=1) from your file.
    Head: NavHead (speed, steer).
    Forward returns (speed[N], steer[N]) as 1D tensors.
    """
    def __init__(
        self,
        pretrained: bool = False,
        normalize: bool = True,
        head_hidden: int = 256,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        # Keep encoder structure and behavior identical to your ResnetEncoder.
        self.encoder = ResnetEncoder(34, pretrained, num_input_images=1)
        self.normalize = normalize
        self.nav = NavHead(in_channels=512, hidden=head_hidden, p_drop=head_dropout)
    def forward(self, x: torch.Tensor):
        # If your inputs might be grayscale [N,1,H,W], handle upstream or repeat to 3ch before calling.
        feats = self.encoder(x, normalize=self.normalize)  # returns 5 features unchanged
        speed, steer = self.nav(feats)
        return speed.squeeze(-1), steer.squeeze(-1)