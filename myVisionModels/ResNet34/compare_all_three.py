import torch

# === Paths ===
ppgeo_path = "resnet34.ckpt"
custom_unstripped_path = "epoch=19-last-custom-ppgeo-trial1.ckpt"
custom_stripped_path = "epoch=19-last-custom-ppgeo-trial1-stripped.ckpt"

def load_state_dict(path, strip_depth_encoder_prefix=False):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    if strip_depth_encoder_prefix:
        new_state = {}
        for k, v in state.items():
            if k.startswith("model.depth_encoder.encoder."):
                # Remove prefix for depth encoder so it matches PPGeo default naming
                new_key = k.replace("model.depth_encoder.encoder.", "")
                new_state[new_key] = v
            else:
                # Keep everything else as-is (pose encoder, motion net, etc.)
                new_state[k] = v
        return new_state
    return state

# Load all three
ppgeo_state = load_state_dict(ppgeo_path)  # default
custom_unstripped_state = load_state_dict(custom_unstripped_path, strip_depth_encoder_prefix=True)
custom_stripped_state = load_state_dict(custom_stripped_path)  # already stripped

def compare_models(name_a, state_a, name_b, state_b):
    keys_a = set(state_a.keys())
    keys_b = set(state_b.keys())

    in_both = sorted(keys_a & keys_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    shape_diffs = [
        (k, state_a[k].shape, state_b[k].shape)
        for k in in_both
        if state_a[k].shape != state_b[k].shape
    ]

    print(f"\n=== {name_a} vs {name_b} ===")
    print(f"✅ Keys in both: {len(in_both)}")
    print(f"❌ Keys only in {name_a}: {len(only_a)}")
    print(f"❌ Keys only in {name_b}: {len(only_b)}")
    print(f"🔍 Shape differences: {len(shape_diffs)}")

    if only_a:
        print(f"   All keys that are only in {name_a}: {only_a}")
    if only_b:
        print(f"   Examples of only in {name_b}: {only_b[:5]}")
    if shape_diffs:
        print(f"   Example shape diff: {shape_diffs[:3]}")

# === Pairwise comparisons ===
compare_models("PPGeo Default", ppgeo_state, "Custom Unstripped (depth encoder only)", custom_unstripped_state)
compare_models("PPGeo Default", ppgeo_state, "Custom Stripped", custom_stripped_state)
compare_models("Custom Unstripped (depth encoder only)", custom_unstripped_state, "Custom Stripped", custom_stripped_state)
