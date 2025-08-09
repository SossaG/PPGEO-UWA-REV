import torch

# Paths to your checkpoints
ppgeo_path = "resnet34.ckpt"
custom_ppgeo_path = "epoch=19-last-custom-ppgeo-trial1-stripped.ckpt"

# Load
ppgeo_ckpt = torch.load(ppgeo_path, map_location='cpu')
ppgeo_state = ppgeo_ckpt['state_dict']

custom_ckpt = torch.load(custom_ppgeo_path, map_location='cpu')
custom_state = custom_ckpt['state_dict'] if 'state_dict' in custom_ckpt else custom_ckpt



# === Compare ===
ppgeo_keys = set(ppgeo_state.keys())
custom_keys = set(custom_state.keys())

only_in_ppgeo = sorted(ppgeo_keys - custom_keys)
only_in_custom = sorted(custom_keys - ppgeo_keys)
in_both = sorted(ppgeo_keys & custom_keys)

shape_diffs = []
for k in in_both:
    if ppgeo_state[k].shape != custom_state[k].shape:
        shape_diffs.append((k, ppgeo_state[k].shape, custom_state[k].shape))

print(f"✅ Keys in both: {len(in_both)}")
print(f"❌ Keys only in ppgeo: {len(only_in_ppgeo)} -> {only_in_ppgeo}")
print(f"❌ Keys only in custom_ppgeo: {len(only_in_custom)} -> {only_in_custom}")
print(f"🔍 Keys with shape differences: {len(shape_diffs)}")
for k, s1, s2 in shape_diffs:
    print(f"  - {k}: ppgeo {s1} vs custom {s2}")
