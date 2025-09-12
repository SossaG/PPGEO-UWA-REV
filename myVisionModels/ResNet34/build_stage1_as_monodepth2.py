# build_stage1_as_monodepth2.py
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

def build_monodepth2_pair_from_stage1(stage1_ckpt_path, device=None):
    """
    Returns (encoder, depth_decoder) ready for Monodepth2-style inference.
    - Fixes the key prefixes so your depth-encoder weights load into ResnetEncoder.
    - Loads your depth-decoder weights directly (they already match).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(stage1_ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)

    # 1) carve out depth encoder/decoder subdicts from the Lightning bundle
    enc_sd_raw = {k.replace("model.depth_encoder.encoder.", ""): v
                  for k, v in sd.items() if k.startswith("model.depth_encoder.encoder.")}
    dec_sd = {k.replace("model.depth_decoder.", ""): v
              for k, v in sd.items() if k.startswith("model.depth_decoder.")}

    # 2) Monodepth2 ResnetEncoder expects keys prefixed with 'encoder.'
    enc_sd = {"encoder." + k: v for k, v in enc_sd_raw.items()}

    # 3) build modules exactly like Monodepth2 test_simple
    enc = ResnetEncoder(num_layers=18, pretrained=False).to(device).eval()
    dec = DepthDecoder(num_ch_enc=enc.num_ch_enc, scales=range(4)).to(device).eval()

    # 4) load weights (strict=True should pass after prefix fix; relax if needed)
    enc.load_state_dict(enc_sd, strict=True)
    dec.load_state_dict(dec_sd, strict=True)

    return enc, dec
