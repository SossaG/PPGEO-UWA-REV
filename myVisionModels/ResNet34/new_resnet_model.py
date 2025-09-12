import torch
import torch.nn as nn
import torchvision.models as tv
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict


# -------- Encoder (PPGeo/Monodepth2 style, fixed grayscale) --------
class PPGeoResnet34EncoderGray(nn.Module):
    """
    ResNet-34 encoder that returns a feature pyramid:
        [relu1, layer1, layer2, layer3, layer4]
    Grayscale ONLY: conv1 expects 1 input channel.
    """
    def __init__(self, normalize: bool = False):
        super().__init__()
        self.normalize = normalize

        backbone = tv.resnet34(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # ImageNet stats (we’ll broadcast channel 0 when normalize=True)
        self.register_buffer("mean0", torch.tensor(0.485).view(1, 1, 1, 1), persistent=False)
        self.register_buffer("std0",  torch.tensor(0.229).view(1, 1, 1, 1), persistent=False)

    @torch.no_grad()
    def load_ppgeo_weights(
        self,
        state: Dict[str, torch.Tensor],
        strict: bool = False,
    ) -> Tuple[set, set]:
        """
        Load PPGeo/ResNet34 weights into grayscale conv trunk.
        - Strips common prefixes.
        - If checkpoint conv1 is RGB (3-ch), average -> 1-ch.
        """
        # unwrap if nested
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]

        def strip_prefix(k: str) -> str:
            for p in ["encoder.", "backbone.", "module.", "encoder.encoder.", "model."]:
                if k.startswith(p):
                    k = k[len(p):]
            return k

        flat = OrderedDict()
        for k, v in state.items():
            sk = strip_prefix(k)
            if sk.split(".")[0] in {"conv1", "bn1", "layer1", "layer2", "layer3", "layer4"}:
                flat[sk] = v

        # conv1: handle 3->1
        if "conv1.weight" in flat and flat["conv1.weight"].dim() == 4:
            w = flat["conv1.weight"]               # [64, C, 7, 7]
            if w.shape[1] == 3:                    # RGB ckpt -> average to gray
                flat["conv1.weight"] = w.mean(1, keepdim=True)

        missing, unexpected = self.load_state_dict(flat, strict=strict)
        loaded_keys = set(flat.keys()) - set(missing)
        return loaded_keys, set(missing)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.normalize:
            x = (x - self.mean0) / self.std0

        x = self.conv1(x)
        x = self.bn1(x)
        relu1 = self.relu(x)
        l1 = self.layer1(self.maxpool(relu1))
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        return [relu1, l1, l2, l3, l4]


# -------- Lightweight navigation head (speed & steering) --------
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
        nn.init.zeros_(self.speed.weight); nn.init.zeros_(self.speed.bias)
        nn.init.zeros_(self.steer.weight); nn.init.zeros_(self.steer.bias)

    def forward(self, feats: List[torch.Tensor]):
        x = self.pool(feats[-1]).flatten(1)
        x = self.mlp(x)
        return self.speed(x), self.steer(x)


# -------- Full model --------
class PPGeoNavModelGray(nn.Module):
    """
    End-to-end: PPGeo-style ResNet34 encoder (grayscale) + 2-branch regressor.
    Forward returns (speed[N], steering[N]).
    """
    def __init__(self, normalize: bool = False, head_hidden: int = 256, head_dropout: float = 0.2):
        super().__init__()
        self.encoder = PPGeoResnet34EncoderGray(normalize=normalize)
        self.head = NavHead(in_channels=512, hidden=head_hidden, p_drop=head_dropout)

    @torch.no_grad()
    def load_ppgeo_checkpoint(
        self,
        ckpt_path: str,
        strict: bool = False,
        map_location: str = "cpu",
    ) -> None:
        state = torch.load(ckpt_path, map_location=map_location)
        loaded, missing = self.encoder.load_ppgeo_weights(state=state, strict=strict)
        print(f"[PPGeoNavModelGray] Loaded {len(loaded)} encoder keys; missing: {len(missing)}")

    def freeze_encoder(self, mode: str = "frozen") -> None:
        """
        mode ∈ {'frozen','partial','unfrozen'}
        'partial' = train conv1+bn1+layer1 only.
        """
        if mode == "frozen":
            for p in self.encoder.parameters():
                p.requires_grad = False
        elif mode == "partial":
            for name, p in self.encoder.named_parameters():
                p.requires_grad = name.startswith(("conv1", "bn1", "layer1"))
        elif mode == "unfrozen":
            for p in self.encoder.parameters():
                p.requires_grad = True
        else:
            raise ValueError("freeze mode must be one of: 'frozen','partial','unfrozen'")

    def forward(self, x: torch.Tensor):
        feats = self.encoder(x)
        speed, steering = self.head(feats)
        return speed.squeeze(-1), steering.squeeze(-1)


# -------- Simple factory for your train script --------
def build_model_for_eglinton_gray(
    pretrain_type: str = "ppgeo",       # {"ppgeo","custom_ppgeo","imagenet","scratch"}
    freeze_mode: str = "unfrozen",      # {"frozen","partial","unfrozen"}
    normalize: bool = False,
    ckpt_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> PPGeoNavModelGray:
    model = PPGeoNavModelGray(normalize=normalize)

    if pretrain_type in {"ppgeo", "custom_ppgeo"}:
        if ckpt_path is None:
            raise ValueError("Provide ckpt_path to PPGeo/custom_ppgeo weights.")
        model.load_ppgeo_checkpoint(ckpt_path=ckpt_path)
    elif pretrain_type == "imagenet":
        # Use torchvision ImageNet weights to initialise conv trunk, then average conv1 to gray
        tv34 = tv.resnet34(weights=tv.ResNet34_Weights.IMAGENET1K_V1)
        # copy conv1 as mean across RGB
        w3 = tv34.conv1.weight.data           # [64,3,7,7]
        model.encoder.conv1.weight.data.copy_(w3.mean(1, keepdim=True))
        # copy remaining blocks cautiously
        for name in ["bn1", "layer1", "layer2", "layer3", "layer4"]:
            getattr(model.encoder, name).load_state_dict(getattr(tv34, name).state_dict())
    elif pretrain_type == "scratch":
        pass
    else:
        raise ValueError("pretrain_type must be one of {'ppgeo','custom_ppgeo','imagenet','scratch'}")

    model.freeze_encoder(freeze_mode)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)
