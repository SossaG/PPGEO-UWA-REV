import torch
import sys
device = torch.device("cuda")
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

#----------------------------------------------------------------------------------------------
from build_stage1_as_monodepth2 import build_monodepth2_pair_from_stage1




# Paths to your checkpoint files
default_ckpt_path = "models_to_test/ppgeo_depth.ckpt"
custom_ckpt_path = "models_to_test/stage1_custom_ppgeo3.ckpt"

# Load checkpoints
ckpt_default = torch.load(default_ckpt_path, map_location="cuda")

#use this part only for testing my depth net stuff:
#  Build encoder/decoder exactly like monodepth2 test_simple.py
enc = ResnetEncoder(num_layers=18, pretrained=False).to(device).eval()
dec = DepthDecoder(num_ch_enc=enc.num_ch_enc, scales=range(4)).to(device).eval()

enc.load_state_dict({k: v for k, v in ckpt_default["depth_encoder_state_dict"].items()
                    if k in enc.state_dict()}, strict=False)
dec.load_state_dict({k: v for k, v in ckpt_default["depth_decoder_state_dict"].items()
                    if k in dec.state_dict()}, strict=False)

custom_enc, custom_dec = build_monodepth2_pair_from_stage1(custom_ckpt_path, device=device)

# === State-dict comparison helpers ===
def compare_state_dicts(left_sd, right_sd, left_name="left", right_name="right", rtol=1e-5, atol=1e-6):
    left_keys = set(left_sd.keys())
    right_keys = set(right_sd.keys())

    only_left = sorted(left_keys - right_keys)
    only_right = sorted(right_keys - left_keys)
    common = sorted(left_keys & right_keys)

    shape_mismatches = []
    value_diffs = []

    for k in common:
        ls = tuple(left_sd[k].shape)
        rs = tuple(right_sd[k].shape)
        if ls != rs:
            shape_mismatches.append((k, ls, rs))
        else:
            # compare values with tolerance
            if not torch.allclose(left_sd[k].detach().cpu(), right_sd[k].detach().cpu(), rtol=rtol, atol=atol):
                value_diffs.append(k)

    # ---- Print summary ----
    print("="*88)
    print(f"Comparing [{left_name}] vs [{right_name}]")
    print(f"Total params: {len(left_keys)} vs {len(right_keys)}")
    print(f"Common keys:  {len(common)}")
    print(f"Only in {left_name}:  {len(only_left)}")
    print(f"Only in {right_name}: {len(only_right)}")
    print(f"Shape mismatches:     {len(shape_mismatches)}")
    print(f"Value differences:     {len(value_diffs)} (rtol={rtol}, atol={atol})")

    if only_left:
        print(f"\n-- Keys only in {left_name} --")
        for k in only_left:
            print("   ", k)

    if only_right:
        print(f"\n-- Keys only in {right_name} --")
        for k in only_right:
            print("   ", k)

    if shape_mismatches:
        print("\n-- Shape mismatches --")
        for k, ls, rs in shape_mismatches:
            print(f"   {k}: {ls} vs {rs}")

    if value_diffs:
        print("\n-- Keys with different values (same shape) --")
        for k in value_diffs:
            print("   ", k)

    print("="*88 + "\n")

    return {
        "only_left": only_left,
        "only_right": only_right,
        "shape_mismatches": shape_mismatches,
        "value_diffs": value_diffs,
    }

# === Run comparisons for encoder and decoder ===
enc_default_sd = enc.state_dict()
dec_default_sd = dec.state_dict()

enc_custom_sd = custom_enc.state_dict()
dec_custom_sd = custom_dec.state_dict()

enc_report = compare_state_dicts(enc_default_sd, enc_custom_sd, left_name="default_encoder", right_name="custom_encoder")
dec_report = compare_state_dicts(dec_default_sd, dec_custom_sd, left_name="default_decoder", right_name="custom_decoder")