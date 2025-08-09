import torch

# Paths to your checkpoint files
default_ckpt_path = "resnet34.ckpt"
custom_ckpt_path = "epoch=19-last-custom-ppgeo-trial1-stripped.ckpt"

# Load checkpoints
ckpt_default = torch.load(default_ckpt_path, map_location="cpu")
ckpt_custom = torch.load(custom_ckpt_path, map_location="cpu")

# Handle Lightning's 'state_dict' nesting
state_dict_default = ckpt_default["state_dict"] if "state_dict" in ckpt_default else ckpt_default
state_dict_custom = ckpt_custom["state_dict"] if "state_dict" in ckpt_custom else ckpt_custom

# Get key sets
keys_default = set(state_dict_default.keys())
keys_custom = set(state_dict_custom.keys())

# Compare
print("\n=== Keys in DEFAULT only ===")
for k in sorted(keys_default - keys_custom):
    print(k)

print("\n=== Keys in CUSTOM only ===")
for k in sorted(keys_custom - keys_default):
    print(k)

print("\n=== Keys in BOTH ===")
for k in sorted(keys_default & keys_custom):
    print(k)

print(f"\nDefault key count: {len(keys_default)}")
print(f"Custom key count:  {len(keys_custom)}")
print(f"Overlap count:     {len(keys_default & keys_custom)}")
