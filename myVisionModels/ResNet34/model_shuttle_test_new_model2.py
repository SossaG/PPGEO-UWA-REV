import torch
from torch import nn
from functools import partial

from torchvision import datasets, models, transforms
from PIL import Image

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from new_resnet_model2 import build_model_for_eglinton, PPGeoNavModel


from torch.utils.tensorboard import SummaryWriter

import numpy as np

import matplotlib
matplotlib.use("Agg")          # put this before any other matplotlib imports
import matplotlib.cm as cm

import cv2
import glob
import os
import sys
from os.path import join, exists, dirname, abspath
import matplotlib.cm as cm

# from sklearn.model_selection import train_test_split

# import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR
import argparse
# from torchsummary import summary
# import time

import random


from pytorch_grad_cam import EigenCAM as LibEigenCAM



transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# Keep Eigen-CAM preprocessing identical to inference
saliency_transform = transform


if __name__ == "__main__":
    #load image
    parser = argparse.ArgumentParser("Load model from checkpoint")
    parser.add_argument("--load_model", action="store_true")
    #parser.add_argument("model_name", default="checkpoints/custom_ppgeo_unfrozen_lane_following_finetune_1.0/ResNet34_shuttle_custom_ppgeo_unfrozen_lane_following_finetune_1.0_1_0.0012_0.4518_0.5270.pth", type=str, nargs='?')
    parser.add_argument("model_name", default="finished_models_new/ResNet34_shuttlebus_custom_ppgeo_frozen_lane_following_finetune_1.0.pth", type=str, nargs='?')
    # parser.add_argument("model_fine_name", default="VMamba_shuttle_lane_following_13_0.0003_0.9026_0.8934.pth", type=str, nargs='?')
    """parser.add_argument("model_pullin_name", default="VMamba_shuttle_pullin_16_0.0007_0.9070_0.8310.pth", type=str, nargs='?')
    parser.add_argument("model_reverse_name", default="VMamba_shuttle_reverse_21_0.0004_0.9536_0.9929.pth", type=str, nargs='?')
    parser.add_argument("model_type", default="lane_following", type=str, nargs='?')"""

    args = parser.parse_args()

    # print(args.model_name)
    # print(args.model_type)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    if device == "cuda":
        torch.cuda.manual_seed_all(1234)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('high')

    """model = ResNet34PilotNet()
    #model.load_state_dict(torch.load(args.model_name)['model_state_dict']) --not using a checkpoint dict file yet, so will replace code for now with the following
    state_dict = torch.load(args.model_name)
    model.load_state_dict(state_dict)"""

    #adding flexivbility for type of .pth loaded
    # --- Build the grayscale PPGeo model skeleton, then load the checkpoint ---
    # We don't load PPGeo/ImageNet pretrain here; we load your saved downstream .pth next.
    model = build_model_for_eglinton(
        pretrain_type="scratch",   # important: we're restoring from your .pth, not re-pretraining here
        freeze_mode="unfrozen",
        normalize=False
    ).to(device)

    _ckpt = torch.load(args.model_name, map_location=device)
    if isinstance(_ckpt, dict) and 'model_state_dict' in _ckpt:
        model.load_state_dict(_ckpt['model_state_dict'], strict=True)
        print(f"[INFO] Loaded full checkpoint from {args.model_name}")
    elif isinstance(_ckpt, dict) and 'state_dict' in _ckpt:
        model.load_state_dict(_ckpt['state_dict'], strict=True)
        print(f"[INFO] Loaded checkpoint.state_dict from {args.model_name}")
    else:
        # fallback: assume raw state_dict-like mapping
        model.load_state_dict(_ckpt, strict=True)
        print(f"[INFO] Loaded raw state_dict from {args.model_name}")

    model.eval()
    model.to(device)



    # === Optional: dynamic model switching support (non-breaking) ===
    # Build a list of candidate model files from the same directory as the initial model.
    try:
        _default_dir = os.path.dirname(args.model_name) or "."
        _cands = sorted(glob.glob(os.path.join(_default_dir, "*.pth")) + glob.glob(os.path.join(_default_dir, "*.pt")))
        pt_model_list = [p for p in _cands if os.path.isfile(p)]
    except Exception:
        pt_model_list = [args.model_name]
    if not pt_model_list:
        pt_model_list = [args.model_name]

    # Index into the list; start at current model if present
    pt_idx = 0
    try:
        pt_idx = pt_model_list.index(args.model_name)
    except ValueError:
        pt_idx = 0
    
    def _reload_model_from_path(pt_model_path: str):
        """Reload a PPGeoNavModel from a given checkpoint path."""
        m = build_model_for_eglinton(
            pretrain_type="scratch",
            freeze_mode="unfrozen",
            normalize=False
        ).to(device)

        checkpoint = torch.load(pt_model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            m.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print(f"[INFO] Loaded full checkpoint from {pt_model_path}")
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            m.load_state_dict(checkpoint['state_dict'], strict=True)
            print(f"[INFO] Loaded checkpoint.state_dict from {pt_model_path}")
        else:
            m.load_state_dict(checkpoint, strict=True)
            print(f"[INFO] Loaded raw state_dict from {pt_model_path}")

        m.eval()
        m.to(device)
        return m


    def _last_conv_before_gap_or_fallback(model: nn.Module) -> nn.Module:
        """Prefer the last conv right before GAP (best interpretability); fallback to last Conv2d."""
        # PPGeoNavModel → encoder is ResNet-34; GAP is applied after layer4
        try:
            return model.encoder.layer4[-1].conv2
        except Exception:
            last = None
            for _, m in model.named_modules():
                if isinstance(m, nn.Conv2d):
                    last = m
            if last is None:
                raise RuntimeError("Could not locate a Conv2d layer for Eigen-CAM.")
            return last



    def _compute_eigencam_overlay(model: nn.Module, saliency_img: Image.Image, device: str) -> np.ndarray:
        """
        No-library Eigen-CAM:
        1) hook last conv → activations A ∈ R^{C,Hf,Wf}
        2) zero-center per-channel, SVD on A_flat ∈ R^{C,HW}
        3) take first left singular vector w, cam = w^T A_flat → (Hf,Wf)
        4) normalize, resize to input crop, blend 50/50 with grayscale
        Returns overlay as float RGB in [0,1] with shape (H, W, 3).
        """
        # Build input exactly like your inference path
        x = saliency_transform(saliency_img).unsqueeze(0).to(device)

        # Capture activations from the target conv
        feats = {}
        target = _last_conv_before_gap_or_fallback(model)

        def _fw_hook(_, __, output):
            feats['act'] = output.detach()

        h = target.register_forward_hook(_fw_hook)
        model.eval()
        with torch.no_grad():
            _ = model(x)  # outputs are unused for Eigen-CAM
        h.remove()

        assert 'act' in feats, "Eigen-CAM: forward hook did not capture activations"
        A = feats['act'][0]              # (C, Hf, Wf) on device
        C, Hf, Wf = A.shape

        # Flatten spatial and zero-center per channel; do SVD on CPU for stability
        # --- SVD on zero-centered activations ---
        Af = A.reshape(C, -1).float().cpu()              # (C, HW)
        Af = Af - Af.mean(dim=1, keepdim=True)

        try:
            U, S, Vh = torch.linalg.svd(Af, full_matrices=False)
            w = U[:, 0]                                  # (C,)
            cam_flat = torch.matmul(w, Af)               # (HW,)
        except Exception:
            U, S, V = torch.svd(Af)
            w = U[:, 0]
            cam_flat = torch.matmul(w, Af)

        cam = cam_flat.reshape(Hf, Wf)                   # (Hf, Wf)

        # >>> NEW: sign disambiguation using feature energy (always ≥0)
        energy = (A.float().cpu() ** 2).sum(dim=0)       # (Hf, Wf)
        corr = (cam * energy).mean()                     # scalar
        if corr < 0:
            cam = -cam                                   # flip sign so it aligns with energy
        # <<<

        
        # >>> NEW: make “importance” strictly positive
        # Option 1 (recommended): keep only positive contribution (ReLU)
        cam = torch.clamp(cam, min=0)
        # (Alternative: magnitude-only)
        # cam = cam.abs()
        # <<<

        # normalise [0,1] as you already do
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam_np = cam.numpy()


        # Resize CAM to crop size
        cam_resized = cv2.resize(cam_np, (saliency_img.width, saliency_img.height), interpolation=cv2.INTER_CUBIC)

        # Base grayscale → BGR uint8
        base_bgr = cv2.cvtColor((np.array(saliency_img)).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # CAM → heatmap (BGR uint8) using a perceptually nicer map than JET
        heat_bgr = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

        # Softer blend (reduce washout)
        ALPHA = 0.35
        overlay_bgr = cv2.addWeighted(base_bgr, 1.0 - ALPHA, heat_bgr, ALPHA, 0.0)
        return overlay_bgr

    



    

    # model_fine = VMamba(
    #         patch_size=(20, 20), 
    #         in_chans=1, 
    #         num_classes=200, 
    #         depths=[3, 4], 
    #         dims=[256, 512], 
    #         # =========================
    #         ssm_d_state=32,
    #         ssm_ratio=2.0,
    #         ssm_dt_rank="auto",
    #         ssm_act_layer="gelu",        
    #         ssm_conv=3,
    #         ssm_conv_bias=True,
    #         ssm_drop_rate=0.0, 
    #         ssm_init="v0",
    #         forward_type="v02",
    #         # =========================
    #         mlp_ratio=4.0,
    #         mlp_act_layer="gelu",
    #         mlp_drop_rate=0.0,
    #         gmlp=False,
    #         # =========================
    #         drop_path_rate=0.0, 
    #         patch_norm=True, 
    #         norm_layer="LN", # "BN", "LN2D"
    #         downsample_version = "v2", # "v1", "v2", "v3"
    #         patchembed_version = "v1", # "v1", "v2"
    #         use_checkpoint=False,  
    #         # =========================
    #         posembed=False,
    #         imgsize=(320, 160),
    #         _SS2D=SS2D,
    #         # =========================
    #         device="cuda"
    #     )

    # model_fine.load_state_dict(torch.load(args.model_fine_name, weights_only=True)['model_state_dict'])

    # model_fine.eval()
    # model_fine.to(device)

    """model_pullin = VMamba(
            patch_size=(20, 9), 
            in_chans=1, 
            num_classes=1024, 
            depths=[3, 4], 
            dims=[256, 512], 
            # =========================
            ssm_d_state=32,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="gelu",        
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.2, 
            ssm_init="v0",
            forward_type="v02",
            # =========================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.2,
            gmlp=False,
            # =========================
            drop_path_rate=0.2, 
            patch_norm=True, 
            norm_layer="LN", # "BN", "LN2D"
            downsample_version = "v2", # "v1", "v2", "v3"
            patchembed_version = "v1", # "v1", "v2"
            use_checkpoint=False,  
            # =========================
            posembed=False,
            imgsize=(400, 180),
            _SS2D=SS2D,
            # =========================
            device="cuda"
        )

    model_pullin.load_state_dict(torch.load(args.model_pullin_name, weights_only=True)['model_state_dict'])

    model_pullin.eval()
    model_pullin.to(device)

    model_reverse = VMamba(
            patch_size=(20, 9), 
            in_chans=1, 
            num_classes=1024, 
            depths=[3, 4], 
            dims=[256, 512], 
            # =========================
            ssm_d_state=32,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="gelu",        
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.2, 
            ssm_init="v0",
            forward_type="v02",
            # =========================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.2,
            gmlp=False,
            # =========================
            drop_path_rate=0.2, 
            patch_norm=True, 
            norm_layer="LN", # "BN", "LN2D"
            downsample_version = "v2", # "v1", "v2", "v3"
            patchembed_version = "v1", # "v1", "v2"
            use_checkpoint=False,  
            # =========================
            posembed=False,
            imgsize=(400, 180),
            _SS2D=SS2D,
            # =========================
            device="cuda"
        )

    model_reverse.load_state_dict(torch.load(args.model_reverse_name, weights_only=True)['model_state_dict'])

    model_reverse.eval()
    model_reverse.to(device)"""

    #model_name = args.model_type --hard code for now
    model_name = "lane_following"

    print("models loaded")


    Image_Paths = []
    All_Searchable_Folders = []

    #generic lanefollowing example
    All_Searchable_Folders = [dirname("/media/sim/data/eglinton_datasorting_dual/sorted_eglinton_data/CIL_Dual_Cam_Stage2_B/lane_following/rosbag2_2024_09_03-10_06_24_0_7421-7571")]

    #roundabout example
    #All_Searchable_Folders = [dirname("/media/sim/data/eglinton_datasorting_dual/sorted_eglinton_data/CIL_Dual_Cam_Stage2_First_Half/roundabout_straight/rosbag2_2024_03_16-11_31_12_0_31144-31362")]
   
    # All_Searchable_Folders = [dirname("/home/quirky/Documents/eglinton_datasorting_dual/sorted_eglinton_data/CIL_Dual_Cam_Stage2_B/pullout/")]

    
    # All_Searchable_Folders = [dirname("/home/quirky/Documents/nUWAyModels/modelEvaluation/drives/rosbag_folder/VMamba/")]
    current_file = 10
    current_model = 0
    # print(All_Searchable_Folders)
    
    random_bag = random.randint(0, len(os.listdir(All_Searchable_Folders[0]))-1)
    # print(os.listdir(All_Searchable_Folders[0]))
    # print(os.listdir(category_file))

    while True:
        try:
        
            image_path = os.listdir(All_Searchable_Folders[0])[random_bag]
            # print(image_path)
            # print(join(All_Searchable_Folders[0],image_path))
            # image_path = All_Searchable_Folders
            # print(sorted(os.listdir(image_path)))
            # print(len(os.listdir(image_path[0])))
            # print(current_file)
            selected_bag = join(All_Searchable_Folders[0],image_path)
            # print(sorted(os.listdir(selected_bag), key=lambda x: int(x.split("_")[1])))
            file = sorted(os.listdir(selected_bag), key=lambda x: int(x.split("_")[1]))[current_file]
            previous_file = sorted(os.listdir(selected_bag), key=lambda x: int(x.split("_")[1]))[current_file-6]
            # file = os.listdir(selected_bag)[0]
            # print(file)
            image_path = join(selected_bag, file)
            previous_data_path = join(selected_bag, previous_file)
            # print(image_path)
            # file = sorted(os.listdir(image_path), key=lambda x: int(x.split("_")[1]))[current_file]
            # print(file)

        # try:
            data = np.load(image_path, allow_pickle=True)
            previous_data = np.load(previous_data_path, allow_pickle=True)
        except EOFError:
            current_file += 1
            continue
        except:
            current_file += 1
            continue

        # print("no error")
        # print(previous_data)

        img = data[0]

        if data[8] is None or data[9] is None:
            Speed, Steering_Angle = data[2], data[3]
        else:
            Speed, Steering_Angle = data[8], data[9]

        # if len(previous_data) == 8:
        #     Speed, Steering_Angle = previous_data[2], previous_data[3]
        # elif len(previous_data) == 10:
        #     # img, Speed, Steering_Angle = data[0], data[8], data[9]
        #     Speed, Steering_Angle = previous_data[2], previous_data[3]
        #     # print("using modified")
        # else:
        #     Speed, Steering_Angle = previous_data[1], previous_data[2]

        cropped_img = img[60:, 40:440]
        cropped_img = Image.fromarray(np.uint8(cropped_img), mode='L')

        saliency_img = cropped_img.copy()

        cropped_img = transform(cropped_img)
        cropped_img = rearrange(cropped_img, "c h w -> 1 c h w")
        cropped_img = cropped_img.to(device)

        # print(Speed)
        # print(Steering_Angle)

        target = None

        with torch.inference_mode(), torch.autocast('cuda', enabled=False):

            #pass image to model
            if current_model == 0:
                output1, output2 = model(cropped_img)

            # elif current_model == 1:
            #     output1, output2 = model_fine(cropped_img)
    
            """elif current_model == 2:
                output1, output2 = model_pullin(cropped_img)

            else:
                output1, output2 = model_reverse(cropped_img)"""

        output1 = output1 * 5.4
        output2 = output2 * 0.3
        output1 = output1.detach().cpu().numpy()[0]
        output2 = output2.detach().cpu().numpy()[0]
        # print(Steering_Angle)

        Speed = Speed * 5.4
        Steering_Angle = Steering_Angle * 0.3
        
        print("my model:")
        print(output1)
        print(output2)
        print("ground truth:")
        print(Speed)
        print(Steering_Angle)
        print(f"current driving mode is: {data[4]}")

        #draw ground truth
        # x1 = [160, 160 + ((Steering_Angle * 20) / 0.3)]
        x1 = (210, 220)
        y1 = (int(210 - ((Steering_Angle * 20) / 0.3)), int(220 - ((Speed * 60) / 5.4)))
        # y1 = [190, 190 - ((Speed * 20) / 5.4)]
        
        #draw model output
        # x2 = [180, 180 + ((output2 * 20) / 0.3)]
        # y2 = [190, 190 - ((output1 * 20) / 5.4)]
        x2 = (270, 220)
        y2 = (int(270 - ((output2 * 20) / 0.3)), int(220 - ((output1 * 60) / 5.4)))

        # plt.plot(x1, y1, x2, y2, color="white", linewidth=5)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.line(img, x1, y1, (0, 255, 0), 5)
        cv2.line(img, x2, y2, (255, 0, 255), 5)
        cv2.imshow('ground truth comparison', img)
       

        #display image
        # plt.imshow(img, cmap='gray')

        #use keys to get next image
        # cv2.waitKey(0)
        # print("waiting for key input")
        k = cv2.waitKey(0)
        # print(k)
        if k == 115:

            ###################### saliency straight #########################

            input_tensor = saliency_transform(saliency_img).unsqueeze(0).to(device)
            input_tensor.requires_grad_()
            target = None

            #pass image to model
            if current_model == 0:
                saliency_output1, saliency_output2 = model(input_tensor)
                target = saliency_output1[0, 0] if saliency_output1.dim() == 2 else saliency_output1.squeeze()

                # Backward pass: compute gradients of output w.r.t. input image
                model.zero_grad()
                target.backward()
            # elif current_model == 1:
            #     saliency_output1, saliency_output2 = model_fine(input_tensor)
            #     target = saliency_output1[0, 0] if saliency_output1.dim() == 2 else saliency_output1.squeeze()

            #     # Backward pass: compute gradients of output w.r.t. input image
            #     model_fine.zero_grad()
            #     target.backward()
            """elif current_model == 2:
                saliency_output1, saliency_output2 = model_pullin(input_tensor)
                target = saliency_output1[0, 0] if saliency_output1.dim() == 2 else saliency_output1.squeeze()

                # Backward pass: compute gradients of output w.r.t. input image
                model_pullin.zero_grad()
                target.backward()
            else:
                saliency_output1, saliency_output2 = model_reverse(input_tensor)
                target = saliency_output1[0, 0] if saliency_output1.dim() == 2 else saliency_output1.squeeze()

                # Backward pass: compute gradients of output w.r.t. input image
                model_reverse.zero_grad()
                target.backward()"""
            saliency = input_tensor.grad.data.abs().squeeze(0).squeeze(0).cpu()   # shape: (3, 224, 224)

            # Original image: resized to match input shape and normalized back to [0, 1]
            original_resized = saliency_img
            original_np = np.array(original_resized) / 255.0  # shape: (224, 224), values in [0, 1]

            # Normalize saliency to [0, 1]
            saliency_np = saliency.numpy()
            saliency_norm = (saliency_np - saliency_np.min()) / (saliency_np.max() - saliency_np.min() + 1e-8)

            # Create a heatmap from the saliency map
            heatmap = cm.jet(saliency_norm)[..., :3]  # shape: (224, 224, 3), ignore alpha channel

            # Convert grayscale to RGB for overlay
            original_rgb = np.stack([original_np]*3, axis=-1)

            # Blend heatmap with original image
            overlay = 0.5 * original_rgb + 0.5 * heatmap
            overlay = np.clip(overlay, 0, 1)
            cv2.namedWindow('Linear Saliency', cv2.WINDOW_NORMAL)
            overlay = cv2.resize(overlay, dsize=(480,240), interpolation=cv2.INTER_CUBIC)
            cv2.resizeWindow('Linear Saliency', 500, 300)

            cv2.imshow('Linear Saliency', overlay)


            ###################### saliency turning #########################

            input_tensor2 = saliency_transform(saliency_img).unsqueeze(0).to(device)
            input_tensor2.requires_grad_()
            target2 = None

            #pass image to model
            if current_model == 0:
                saliency_output1, saliency_output2 = model(input_tensor2)
                target2 = saliency_output2[0, 0] if saliency_output2.dim() == 2 else saliency_output2.squeeze()

                # Backward pass: compute gradients of output w.r.t. input image
                model.zero_grad()
                target2.backward()
            # elif current_model == 1:
            #     saliency_output1, saliency_output2 = model_fine(input_tensor2)
            #     target2 = saliency_output2[0, 0] if saliency_output2.dim() == 2 else saliency_output2.squeeze()

            #     # Backward pass: compute gradients of output w.r.t. input image
            #     model_fine.zero_grad()
            #     target2.backward()
            """elif current_model == 2:
                saliency_output1, saliency_output2 = model_pullin(input_tensor2)
                target2 = saliency_output2[0, 0] if saliency_output2.dim() == 2 else saliency_output2.squeeze()

                # Backward pass: compute gradients of output w.r.t. input image
                model_pullin.zero_grad()
                target2.backward()
            else:
                saliency_output1, saliency_output2 = model_reverse(input_tensor2)
                target2 = saliency_output2[0, 0] if saliency_output2.dim() == 2 else saliency_output2.squeeze()

                # Backward pass: compute gradients of output w.r.t. input image
                model_reverse.zero_grad()
                target2.backward()"""
            saliency2 = input_tensor2.grad.data.abs().squeeze(0).squeeze(0).cpu()   # shape: (3, 224, 224)

            # Original image: resized to match input shape and normalized back to [0, 1]
            original_resized2 = saliency_img
            original_np2 = np.array(original_resized2) / 255.0  # shape: (224, 224), values in [0, 1]

            # Normalize saliency to [0, 1]
            saliency_np2 = saliency2.numpy()
            saliency_norm2 = (saliency_np2 - saliency_np2.min()) / (saliency_np2.max() - saliency_np2.min() + 1e-8)

            # Create a heatmap from the saliency map
            heatmap2 = cm.jet(saliency_norm2)[..., :3]  # shape: (224, 224, 3), ignore alpha channel

            # Convert grayscale to RGB for overlay
            original_rgb2 = np.stack([original_np2]*3, axis=-1)

            # Blend heatmap with original image
            overlay2 = 0.5 * original_rgb2 + 0.5 * heatmap2
            overlay2 = np.clip(overlay2, 0, 1)
            cv2.namedWindow('Rotational Saliency', cv2.WINDOW_NORMAL)
            overlay2 = cv2.resize(overlay2, dsize=(480,240), interpolation=cv2.INTER_CUBIC)
            cv2.resizeWindow('Rotational Saliency', 500, 300)

            cv2.imshow('Rotational Saliency', overlay2)


            # plt.figure(figsize=(6, 6))
            # plt.imshow(overlay)
            # plt.axis("off")
            # plt.title("Saliency Map Overlay")
            # # plt.ion()
            # # plt.show()
            # plt.pause(0.001)
            k = cv2.waitKey(0)


        

        if k == 113:
            cv2.destroyAllWindows()
            break
        elif k == 83:
            current_file += 1
            if current_file > (len(os.listdir(selected_bag))-1):
                current_file = (len(os.listdir(selected_bag))-1)
            continue
        elif k == 81:
            current_file -= 1
            if current_file < 10:
                current_file = 10
            continue
        elif k == 82:
            current_file += 20
            if current_file > (len(os.listdir(selected_bag))-1):
                current_file = (len(os.listdir(selected_bag))-1)
            continue
        elif k == 84:
            current_file -= 20
            if current_file < 10:
                current_file = 10
            continue
        elif k == 114:
            current_model = 3
            continue
        # elif k == 102:
        #     current_model = 1
            continue
        elif k == 112:
            current_model = 2
            continue
        elif k == 108:
            current_model = 0
            continue
        elif k == 99:
            # random_category = random.randint(0, len(All_Searchable_Folders)-1)
            # category_file = join(All_Searchable_Folders, All_Searchable_Folders[random_category])
            # random_bag = random.randint(0, len(os.listdir(category_file))-1)
            random_bag = random.randint(0, len(os.listdir(All_Searchable_Folders[0])))
            current_file = 0
            continue
        
        # elif k == :
        #     random_bag = random.randint(0, len(All_Searchable_Folders))
        #     current_file = 0
        #     continue
        
        elif k == ord('9'):
            # Cycle backward through available PyTorch model checkpoints
            try:
                pt_idx = (pt_idx - 1) % len(pt_model_list)
                selected_model_idx = pt_idx
                pt_model_path = pt_model_list[selected_model_idx]
                model = _reload_model_from_path(pt_model_path)
                print(f"[INFO] Switched to PyTorch model {selected_model_idx + 1}/{len(pt_model_list)}: {pt_model_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load model from {pt_model_path}: {e}")
                continue
        elif k == ord('0'):
            # Cycle forward through available PyTorch model checkpoints
            try:
                pt_idx = (pt_idx + 1) % len(pt_model_list)
                selected_model_idx = pt_idx
                pt_model_path = pt_model_list[selected_model_idx]
                model = _reload_model_from_path(pt_model_path)
                print(f"[INFO] Switched to PyTorch model {selected_model_idx + 1}/{len(pt_model_list)}: {pt_model_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load model from {pt_model_path}: {e}")
                continue
        else:
            pass