import torch
import os

# ✏️ EDIT THIS to point to your checkpoint or state_dict file
ckpt_path = "finished_models_new_final/ResNet34_shuttlebus_custom_ppgeo_frozen_lane_following_finetune_0.01.pth"

def is_state_dict_only(ckpt_data):
    return isinstance(ckpt_data, dict) and all(isinstance(v, torch.Tensor) for v in ckpt_data.values())

def extract_state_dict(ckpt_data):
    if is_state_dict_only(ckpt_data):
        print("[Info] Detected RAW state_dict (not a full checkpoint).")
        return ckpt_data
    elif isinstance(ckpt_data, dict):
        print("[Info] Detected full checkpoint file.")
        print(f"[Checkpoint Keys] {list(ckpt_data.keys())}")

        if "state_dict" in ckpt_data:
            print("[Info] Using 'state_dict' key.")
            return ckpt_data["state_dict"]

        # Try to auto-select subkey that looks like a state dict
        for k, v in ckpt_data.items():
            if isinstance(v, dict) and all(isinstance(p, torch.Tensor) for p in v.values()):
                print(f"[Info] Using subkey '{k}' as state_dict.")
                return v

        raise ValueError("No valid state_dict found in checkpoint.")
    else:
        raise TypeError("Unknown checkpoint format.")

def print_state_dict(state_dict):
    print("\n[State Dict Contents]")
    for name, param in state_dict.items():
        print(f"{name:<60} {tuple(param.shape)}")

def main():
    if not os.path.exists(ckpt_path):
        print(f"[Error] File not found: {ckpt_path}")
        return

    print(f"[Info] Loading checkpoint: {ckpt_path}")
    ckpt_data = torch.load(ckpt_path, map_location="cpu")
    state_dict = extract_state_dict(ckpt_data)
    print_state_dict(state_dict)

if __name__ == "__main__":
    main()
