import torch
from collections import OrderedDict

# ==== Paths ====
ckpt_path = "epoch=19-last-custom-ppgeo-trial1.ckpt"   # your full Lightning ckpt
stripped_path = "epoch=19-last-custom-ppgeo-trial1-stripped.ckpt"  # output stripped weights

# ==== 1. Load the Lightning checkpoint ====
ckpt = torch.load(ckpt_path, map_location="cpu")

if "state_dict" not in ckpt:
    raise ValueError("The checkpoint does not have a 'state_dict' key. Is this a Lightning .ckpt file?")

state_dict = ckpt["state_dict"]

print(f"Original ckpt keys: {len(state_dict)}")

# ==== 2. Keep only depth_encoder.encoder.* ====
stripped_state_dict = OrderedDict()

for k, v in state_dict.items():
    if k.startswith("model.depth_encoder.encoder."):
        new_key = k.replace("model.depth_encoder.encoder.", "")
        stripped_state_dict[new_key] = v

print(f"Stripped keys: {len(stripped_state_dict)}")

# ==== 3. Save stripped weights ====
torch.save(stripped_state_dict, stripped_path)
print(f"✅ Saved stripped encoder weights to {stripped_path}")

# ==== 4. Quick check ====
loaded = torch.load(stripped_path, map_location="cpu")
print(f"Keys in saved file: {len(loaded)}")
print(list(loaded.keys())[:10], "...")
