# try_load_stage1_into_monodepth2.py
import os
import sys

#add monodpeth2 path to sys path so that dont have to import all the dependant scripts manually

monodepth2_dir = "/media/sim/data/ivan/monodepth2"
sys.path.append(monodepth2_dir)
#imports for default ppgeo depth------------------------------------------------
import torch, PIL.Image as pil
import numpy as np
from torchvision import transforms
from matplotlib import cm
from layers import disp_to_depth  
#append path for inner networks dir-------------------------------------------------
monodepth2_networks_dir = "/media/sim/data/ivan/monodepth2/networks"
sys.path.append(monodepth2_networks_dir)
from resnet_encoder import ResnetEncoder                       # :contentReference[oaicite:1]{index=1}
from depth_decoder import DepthDecoder                         # :contentReference[oaicite:2]{index=2}

ckpt = torch.load("models_to_test/stage1_custom_ppgeo.ckpt", map_location="cpu")
sd = ckpt.get("state_dict", ckpt)

# 1) carve out your depth-* subdicts from Lightning bundle
enc_sd = {k.replace("model.depth_encoder.encoder.", ""): v
          for k, v in sd.items() if k.startswith("model.depth_encoder.encoder.")}
dec_sd = {k.replace("model.depth_decoder.", ""): v
          for k, v in sd.items() if k.startswith("model.depth_decoder.")}

# 2) build monodepth2 modules
enc = ResnetEncoder(num_layers=34, pretrained=False)
dec = DepthDecoder(num_ch_enc=enc.num_ch_enc, scales=range(4))

# 3) try loading with strict=False to see compatibility
missing_e, unexpected_e = enc.load_state_dict(enc_sd, strict=False)
missing_d, unexpected_d = dec.load_state_dict(dec_sd, strict=False)

print("\n== Encoder load ==")
print("missing:", [k for k in missing_e][:20], " ... total", len(missing_e))
print("unexpected:", [k for k in unexpected_e][:20], " ... total", len(unexpected_e))

print("\n== Decoder load ==")
print("missing:", [k for k in missing_d][:20], " ... total", len(missing_d))
print("unexpected:", [k for k in unexpected_d][:20], " ... total", len(unexpected_d))

print("\nIf both 'missing' lists are small/empty, you can reuse Monodepth2 test_simple.py as-is.")
