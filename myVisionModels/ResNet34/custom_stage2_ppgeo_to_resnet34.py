import torch

def main():
    # Hardcoded input and output paths
    input_ckpt_path = "models_to_test/stage2_custom_ppgeo2.ckpt"
    output_pt_path = "resnet34_custom2.pt"

    print(f"[Info] Loading checkpoint from: {input_ckpt_path}")
    ckpt = torch.load(input_ckpt_path, map_location="cpu")

    if "state_dict" not in ckpt:
        raise ValueError("Checkpoint does not contain a 'state_dict' key.")

    state_dict = ckpt["state_dict"]

    prefix = "motionnet.visual_encoder.encoder."
    print(f"[Info] Filtering keys with prefix: {prefix}")

    # Filter and strip keys
    resnet_state_dict = {
        k[len(prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(prefix)
    }

    print(f"[Info] Extracted {len(resnet_state_dict)} ResNet keys.")

    # Optional: show a few keys
    for i, k in enumerate(resnet_state_dict.keys()):
        if i < 10:
            print(f"  -> {k}")
        elif i == 10:
            print("  ...")

    # Save stripped state dict
    torch.save(resnet_state_dict, output_pt_path)
    print(f"[Success] Saved converted state_dict to: {output_pt_path}")

if __name__ == "__main__":
    main()
