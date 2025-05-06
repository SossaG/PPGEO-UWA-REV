import torch

# Path to your downloaded PPGeo checkpoint
ckpt_path = 'resnet34.ckpt'  # change this if it's in another folder

# Load the checkpoint
ckpt = torch.load(ckpt_path, map_location='cpu')

# Print top-level info
print("Type of checkpoint:", type(ckpt))
print("Top-level keys:", ckpt.keys() if isinstance(ckpt, dict) else "Not a dict")

# If it has a 'state_dict', print the nested keys
if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    print("\nNested state_dict keys:")
    for k in ckpt['state_dict'].keys():
        print(k)
elif isinstance(ckpt, dict):
    print("\nState_dict keys:")
    for k in ckpt.keys():
        print(k)
else:
    print("This checkpoint format is not expected.")
