# ppgeo_nav_from_ppgeo_encoder.py

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from collections import OrderedDict

#append path for inner networks dir-------------------------------------------------
import sys
monodepth2_networks_dir = "/media/sim/data/ivan/monodepth2/networks"
sys.path.append(monodepth2_networks_dir)
# Import the EXACT PPGeo/Monodepth2-style encoder wrapper you provided
from resnet_encoder import ResnetEncoder         # <- keeps structure identical               # :contentReference[oaicite:1]{index=1}

# ---- Adapter that preserves the PPGeo encoder exactly ----
class PPGeoResnet34EncoderAdapter(nn.Module):
    """
    Thin wrapper around ResnetEncoder(num_layers=34, pretrained=False).
    - Leaves the encoder structure IDENTICAL to PPGeo's definition.
    - If input is grayscale [N,1,H,W], repeats to RGB before forward().
    - Optionally uses the normalize=True path exposed by ResnetEncoder.
    """
    def __init__(self, normalize: bool = False):
        super().__init__()
        self.normalize = normalize
        self.encoder = ResnetEncoder(num_layers=34, pretrained=False, num_input_images=1)

    @torch.no_grad()
    def load_ppgeo_weights(self, state: Dict[str, torch.Tensor], strict: bool = False) -> Tuple[set, set]:
        """
        Load PPGeo/ResNet34 weights into the underlying ResnetEncoder.
        Handles common prefix patterns seen in checkpoints:
          - 'state_dict.' prefix
          - 'encoder.encoder.' -> 'encoder.'
          - plain torchvision resnet keys (e.g., 'layer1.*') by prefixing 'encoder.'
        """
        sd = state
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]

        def strip_prefix(k: str) -> str:
            # remove a leading "state_dict." if present
            if k.startswith("state_dict."):
                k = k[len("state_dict."):]
            return k

        flat = OrderedDict()
        for k, v in sd.items():
            k = strip_prefix(k)

            # Normalize key space:
            # - If keys look like 'encoder.encoder.layerX...', collapse to 'encoder.layerX...'
            if k.startswith("encoder.encoder."):
                k = "encoder." + k[len("encoder.encoder."):]
            # - If keys start with bare resnet trunk (e.g., 'layer1.0.conv1.weight'), prepend 'encoder.'
            elif k.split(".")[0] in {"conv1", "bn1", "layer1", "layer2", "layer3", "layer4"} and not k.startswith("encoder."):
                k = "encoder." + k

            # Only pass through parameters that exist under the PPGeo trunk
            if k.startswith(("encoder.conv1", "encoder.bn1", "encoder.layer1", "encoder.layer2", "encoder.layer3", "encoder.layer4")):
                flat[k] = v

        missing, unexpected = self.encoder.load_state_dict(flat, strict=strict)
        loaded_keys = set(flat.keys()) - set(missing)
        return loaded_keys, set(missing)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Keep encoder IDENTICAL; if grayscale, convert to 3ch by repeat
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # Use the encoder's native 'normalize' switch for ImageNet stats if desired
        feats = self.encoder(x, normalize=self.normalize)
        # feats is [relu1, layer1, layer2, layer3, layer4] exactly as PPGeo returns
        return feats


# ---- Lightweight dual-regression head ----
class NavHead(nn.Module):
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
        # small, neutral init
        nn.init.zeros_(self.speed.weight); nn.init.zeros_(self.speed.bias)
        nn.init.zeros_(self.steer.weight); nn.init.zeros_(self.steer.bias)

    def forward(self, feats: List[torch.Tensor]):
        x = self.pool(feats[-1]).flatten(1)  # 512 feature map from layer4
        x = self.mlp(x)
        return self.speed(x), self.steer(x)


# ---- Full nav model ----
class PPGeoNavModel(nn.Module):
    """
    Encoder: EXACT PPGeo ResnetEncoder(34) (3-ch) via adapter.
    Head: two regressors (speed, steering).
    Forward returns (speed[N], steering[N]).
    """
    def __init__(self, normalize: bool = False, head_hidden: int = 256, head_dropout: float = 0.2):
        super().__init__()
        self.encoder = PPGeoResnet34EncoderAdapter(normalize=normalize)
        self.head = NavHead(in_channels=512, hidden=head_hidden, p_drop=head_dropout)

    @torch.no_grad()
    def load_ppgeo_checkpoint(self, ckpt_path: str, strict: bool = False, map_location: str = "cpu") -> None:
        state = torch.load(ckpt_path, map_location=map_location)
        loaded, missing = self.encoder.load_ppgeo_weights(state=state, strict=strict)
        print(f"[PPGeoNavModel] Loaded {len(loaded)} encoder keys; missing: {len(missing)}")

    def freeze_encoder(self, mode: str = "frozen") -> None:
        """
        mode ∈ {'frozen','partial','unfrozen'}
        'partial' trains only early layers (conv1/bn1/layer1) similar to common practice.
        """
        if mode == "frozen":
            for p in self.encoder.parameters():
                p.requires_grad = False
        elif mode == "partial":
            for name, p in self.encoder.named_parameters():
                p.requires_grad = name.startswith(("encoder.encoder.conv1",
                                                   "encoder.encoder.bn1",
                                                   "encoder.encoder.layer1",
                                                   "encoder.conv1", "encoder.bn1", "encoder.layer1"))
        elif mode == "unfrozen":
            for p in self.encoder.parameters():
                p.requires_grad = True
        else:
            raise ValueError("freeze mode must be one of: 'frozen','partial','unfrozen'")

    def forward(self, x: torch.Tensor):
        feats = self.encoder(x)            # exact PPGeo feature pyramid
        speed, steering = self.head(feats) # pooled from layer4
        return speed.squeeze(-1), steering.squeeze(-1)


# ---- Small factory to match your train script usage ----
def build_model_for_eglinton(
    pretrain_type: str = "ppgeo",       # {"ppgeo","imagenet","scratch"}
    freeze_mode: str = "unfrozen",      # {"frozen","partial","unfrozen"}
    normalize: bool = False,
    ckpt_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> PPGeoNavModel:
    model = PPGeoNavModel(normalize=normalize)

    if pretrain_type == "ppgeo":
        if ckpt_path is None:
            raise ValueError("Provide ckpt_path to PPGeo weights (e.g., 'resnet34.ckpt').")
        model.load_ppgeo_checkpoint(ckpt_path=ckpt_path)
    elif pretrain_type == "imagenet":
        # optional: initialise via torchvision IMAGENET and load into encoder.encoder.* subtree
        import torchvision as tv
        tv34 = tv.models.resnet34(weights=tv.models.ResNet34_Weights.IMAGENET1K_V1)
        # load straight into underlying torchvision trunk params
        model.encoder.encoder.encoder.load_state_dict(tv34.state_dict(), strict=False)
    elif pretrain_type == "scratch":
        pass
    else:
        raise ValueError("pretrain_type must be one of {'ppgeo','imagenet','scratch'}")

    model.freeze_encoder(freeze_mode)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)
