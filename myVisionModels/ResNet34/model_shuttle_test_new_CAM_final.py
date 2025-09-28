import torch
from torch import nn
from functools import partial

from torchvision import datasets, models, transforms
from PIL import Image
import PIL.Image as PILImage  # robust isinstance target

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from new_resnet_model_final import EglintonNavModel


from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import numpy as np

import cv2
import os
import sys
from os.path import join, exists, dirname, abspath
import matplotlib.cm as cm
import glob

# from sklearn.model_selection import train_test_split

# import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR
import argparse
# from torchsummary import summary
# import time

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

saliency_transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def eigen_cam_from_feats(feat: torch.Tensor) -> torch.Tensor:
    """
    Input:
      feat: [C, H, W] tensor from encoder's last stage (layer4), no grad.
    Returns:
      cam: [H, W] tensor in [0,1], same spatial size as feat
    """
    C, H, W = feat.shape
    # Flatten spatial dims: M shape [C, HW]
    M = feat.view(C, -1)  # [C, HW]

    # SVD on channel covariance: principal eigenvector (U[:,0]) gives channel weights
    # M = U S V^T ; principal weights w = U[:,0]
    # Torch SVD is deterministic with float32 on CPU/GPU for this size.
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)  # U: [C, C], Vh: [HW, HW]
    w = U[:, 0]  # [C]

    # Combine channels with weights to get spatial map: cam_flat = w^T @ M
    cam_flat = torch.mv(M.transpose(0, 1), w)  # [HW]
    cam = cam_flat.view(H, W)

    # Normalise to [0,1]
    cam = cam - cam.min()
    denom = cam.max().clamp(min=1e-6)
    cam = cam / denom
    return cam

@torch.no_grad()
def eigen_cam_on_grayscale(
    model: EglintonNavModel,
    saliency_img,  
    device: str = None,
    return_overlay: bool = False
):
    """
    Computes Eigen-CAM heatmap for your Eglinton model's encoder on a single grayscale image.

    Args:
      model: EglintonNavModel (already constructed; weights optional)
      saliency_img: grayscale image HxW (uint8 0..255 or float 0..255/0..1)
      device: e.g. 'cuda' or 'cpu' (defaults to model's device)
      return_overlay: if True, also returns an RGB overlay [H,W,3] numpy
    Returns:
      cam_np: [H, W] float32 in [0,1]
      (optional) overlay_rgb: [H, W, 3] uint8
    """
    model.eval()
    # Pick device from model if not provided
    if device is None:
        device = next(model.parameters()).device.type

    arr = np.asarray(saliency_img, dtype=np.float32)  # [H,W], values 0..255
    arr = arr / 255.0                                # scale to [0,1]
    img_t = torch.from_numpy(arr)[None, None, ...]   # [1,1,H,W]
    img_t = img_t.to(device)



    # --- Run encoder to get multi-scale features (your encoder returns 5) ---
    # Your EglintonNavModel stores normalize flag and passes it to ResnetEncoder,
    # which internally uses mean=0.458, std=0.245 for grayscale normalization.
    feats = model.encoder(img_t, normalize=model.normalize)
    last = feats[-1].squeeze(0)  # [C,Hf,Wf]

    # --- Eigen-CAM on last feature map ---
    cam = eigen_cam_from_feats(last)  # [Hf,Wf]

    # --- Upsample to input image size ---
    H, W = img_t.shape[-2:]
    cam_up = F.interpolate(cam[None, None, ...], size=(H, W), mode="bilinear", align_corners=False)
    cam_up = cam_up[0, 0]  # [H,W]

    # Normalise again after interpolation
    cam_up = cam_up - cam_up.min()
    cam_up = cam_up / cam_up.max().clamp(min=1e-6)

    cam_np = cam_up.detach().cpu().float().numpy()

    if not return_overlay:
        return cam_np

    # Simple overlay (grayscale base + CAM as alpha tint)
    base = (img_t[0, 0].detach().cpu().clamp(0, 1).numpy() * 255.0).astype(np.uint8)  # [H,W]
    heat = (cam_np * 255.0).astype(np.uint8)  # [H,W]

    # Make a quick jet-like overlay without cv2: stack channels (R=heat, G=base, B=(255-heat))
    overlay = np.stack([
        heat,                                # R
        base,                                # G
        255 - heat                           # B
    ], axis=-1).astype(np.uint8)             # [H,W,3]

    return cam_np, overlay

def show_eigencam_overlay_window(
    saliency_img,          # HxW grayscale (uint8 0..255 or float 0..255/0..1)
    model,                 # your EglintonNavModel
    win_name="saliency",
    target_size=(480, 240) # (W,H) for display
):
    # --- Run Eigen-CAM ---
    cam, _ = eigen_cam_on_grayscale(model, saliency_img, return_overlay=True)  # cam: [H,W] float [0,1]

    # --- Build heatmap (like your old code) ---
    # cv2 expects uint8 0..255 and BGR ordering
    cam_uint8 = (cam * 255.0).astype(np.uint8)                    # [H,W]
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)  # [H,W,3] uint8 (BGR)

    # --- Prepare original grayscale image as 3-channel for blending ---
    if isinstance(saliency_img, np.ndarray):
        orig = saliency_img.astype(np.float32)
    else:
        orig = np.asarray(saliency_img, dtype=np.float32)

    if orig.max() > 1.0:
        orig = orig / 255.0  # now 0..1

    # make it 3-channel (BGR for cv2)
    original_bgr = np.stack([orig, orig, orig], axis=-1)  # [H,W,3], float 0..1

    # scale heatmap to 0..1 for blending
    heatmap_bgr_float = heatmap_bgr.astype(np.float32) / 255.0

    # --- Blend like your snippet ---
    overlay = 0.5 * original_bgr + 0.5 * heatmap_bgr_float
    overlay = np.clip(overlay, 0.0, 1.0)

    # --- Resize for display (note: cv2.resize uses (W,H)) ---
    overlay_disp = cv2.resize(overlay, dsize=target_size, interpolation=cv2.INTER_CUBIC)

    # --- Show window, keep previous window behaviour ---
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 500, 300)
    cv2.imshow(win_name, overlay_disp)
    # caller should handle cv2.waitKey(...) outside for loop timing/controls

    return overlay_disp  # float image 0..1 (useful if you want to save or log)




if __name__ == "__main__":
    #load image
    parser = argparse.ArgumentParser("Load model from checkpoint")
    parser.add_argument("--load_model", action="store_true")
    #parser.add_argument("model_name", default="finished_models_new_final/ResNet34_shuttlebus_custom_ppgeo_frozen_lane_following_finetune_1.0.pth", type=str, nargs='?')
    #parser.add_argument("model_name", default="checkpoints_new_final/ppgeo_frozen_lane_following_finetune_1.0/ResNet34_shuttle_ppgeo_frozen_lane_following_finetune_1.0_3_0.0015_0.3584_0.3335.pth", type=str, nargs='?')
    parser.add_argument("model_name", default="checkpoints_new_final/imagenet_frozen_lane_following_finetune_0.01/ResNet34_shuttle_imagenet_frozen_lane_following_finetune_0.01_2_0.0061_0.0135_0.1631.pth", type=str, nargs='?')
    """parser.add_argument("model_fine_name", default="VMamba_shuttle_lane_following_finetune_7_0.0003_0.9266_0.8902.pth", type=str, nargs='?')
    parser.add_argument("model_pullin_name", default="VMamba_shuttle_pullin_11_0.0003_0.9749_0.9469.pth", type=str, nargs='?')
    parser.add_argument("model_reverse_name", default="VMamba_shuttle_reverse_11_0.0002_0.9605_0.9967.pth", type=str, nargs='?')
    parser.add_argument("model_type", default="lane_following", type=str, nargs='?')"""

    args = parser.parse_args()

    print(args.model_name)
    #print(args.model_type)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

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

    """torch.manual_seed(1234)
    if device == "cuda":
        torch.cuda.manual_seed_all(1234)"""

    model = EglintonNavModel(pretrained=True, normalize=True)
    _ckpt = torch.load(args.model_name, map_location=device)
    if isinstance(_ckpt, dict):
        # common keys used in saved checkpoints
        if 'model_state_dict' in _ckpt:
            model.load_state_dict(_ckpt['model_state_dict'])
            print(f"[INFO] Loaded full checkpoint from {args.model_name}")
        elif 'state_dict' in _ckpt:
            model.load_state_dict(_ckpt['state_dict'])
            print(f"[INFO] Loaded checkpoint.state_dict from {args.model_name}")
        else:
            # fallback: some people save the raw sd inside a single-key dict
            try:
                # try the only tensor dict inside
                possible = {k: v for k, v in _ckpt.items() if isinstance(v, torch.Tensor)}
                if possible:
                    model.load_state_dict(_ckpt)
                    print(f"[INFO] Loaded dict as raw state_dict from {args.model_name}")
                else:
                    # last resort: assume it's already a state_dict-like mapping
                    model.load_state_dict(_ckpt)
                    print(f"[WARN] Unknown checkpoint format; attempted direct load: {args.model_name}")
            except Exception as e:
                raise RuntimeError(f"Unrecognised checkpoint format at {args.model_name}: {e}")
    else:
        # raw state_dict tensor mapping
        model.load_state_dict(_ckpt)
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
        """Reload a ResNet34PilotNet from a given checkpoint path."""
        m = EglintonNavModel(pretrained=True, normalize=True)
        checkpoint = torch.load(pt_model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            m.load_state_dict(checkpoint['model_state_dict'])
            print(f"[INFO] Loaded full checkpoint from {pt_model_path}")
        else:
            m.load_state_dict(checkpoint)
            print(f"[INFO] Loaded raw state_dict from {pt_model_path}")
        m.eval()
        m.to(device)
        return m






    



    #model_name = args.model_type --hard code for now
    model_name = "lane_following"

    print("models loaded")


    Image_Paths = []
    All_Searchable_Folders = []

    #generic lanefollowing example
    #All_Searchable_Folders = [dirname("/media/sim/data/eglinton_datasorting_dual/sorted_eglinton_data/CIL_Dual_Cam_Stage2_B/lane_following/rosbag2_2024_09_03-10_06_24_0_7421-7571")]
    All_Searchable_Folders = [dirname("/media/sim/data/eglinton_datasorting_dual/sorted_eglinton_data/CIL_Dual_Cam_Stage2_B/lane_following/rosbag2_2025_02_21-14_05_53_0")]
    #roundabout example
    #All_Searchable_Folders = [dirname("/media/sim/data/eglinton_datasorting_dual/sorted_eglinton_data/CIL_Dual_Cam_Stage2_First_Half/roundabout_straight/rosbag2_2024_03_16-11_31_12_0_31144-31362")]


   #pullin stops example
    #All_Searchable_Folders = [dirname("/media/sim/data/eglinton_datasorting_dual/sorted_eglinton_data/CIL_Dual_Cam_Stage2_B/pullin_stops/rosbag2_2025_02_21-14_45_21_0_51-2250_stops")]

  #startpoint pull out eg
    #All_Searchable_Folders = [dirname("/media/sim/data/eglinton_datasorting_dual/sorted_eglinton_data/CIL_Dual_Cam_Stage2_B/startpoint_out/rosbag2_2025_01_22-11_37_06_0_826-1251")]
    #All_Searchable_Folders = [dirname("/home/quirky/Documents/eglinton_datasorting_dual/sorted_eglinton_data/CIL_Dual_Cam_Stage2_B/pullout/")]

    
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
            show_eigencam_overlay_window(saliency_img, model, win_name="saliency", target_size=(480,240))


            
            k = cv2.waitKey(0)
            print(k)

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