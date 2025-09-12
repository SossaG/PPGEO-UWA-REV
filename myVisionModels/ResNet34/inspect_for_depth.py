# inspect_for_depth.py  (updated)
import argparse, os, json, re, sys
from collections import Counter
import torch

def try_load(path, allow_pickle=False):
    # 1) safest: weights_only=True
    try:
        return torch.load(path, map_location="cpu", weights_only=True), "weights_only"
    except Exception as e:
        if not allow_pickle:
            return {"__load_error__": str(e)}, "failed_weights_only"
    # 2) trusted fallback: full pickle (requires trusting the file!)
    try:
        # Import lightning if available to satisfy class references
        try:
            import lightning  # noqa: F401
        except Exception:
            try:
                import pytorch_lightning  # noqa: F401
            except Exception:
                pass
        return torch.load(path, map_location="cpu"), "pickle"
    except Exception as e:
        return {"__load_error__": str(e)}, "failed_pickle"

def is_state_dict_like(obj):
    return isinstance(obj, dict) and any(torch.is_tensor(v) for v in obj.values())

def extract_state_dict(obj):
    # Lightning: top-level 'state_dict'
    if isinstance(obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "weights"]:
            if k in obj and is_state_dict_like(obj[k]):
                meta = {kk: vv for kk, vv in obj.items() if kk != k}
                return obj[k], meta
        if is_state_dict_like(obj):
            return obj, {}
    if hasattr(obj, "state_dict"):
        try:
            return obj.state_dict(), {}
        except Exception:
            pass
    return None, {}

def top_prefixes(keys, depth=2, n=20):
    c = Counter(".".join(k.split(".")[:depth]) for k in keys)
    return [{"prefix": p, "count": c[p]} for p in [k for k, _ in c.most_common(n)]]

def guess_arch(sd, meta):
    keys = list(sd.keys())
    j = lambda pat: any(re.search(pat, k) for k in keys)
    if any(k in meta for k in ["height","width"]) and j(r"(^|\.)(layer1|layer2|layer3|layer4)\."):
        return "Monodepth2_ResNet_Encoder"
    if j(r"\bconvs\.(upconv|dispconv)") and j(r"\bdispconv"):
        return "Monodepth2_DepthDecoder"
    if j(r"\bconvs\.squeeze\b") and j(r"\bconvs\.pose\.0\b"):
        return "Monodepth2_PoseDecoder"
    if j(r"\bconvs\.0\.weight\b") and j(r"\bpose_conv\.weight\b"):
        return "Monodepth2_PoseCNN"
    # bundle?
    has_enc = j(r"(^|\.)(encoder|layer1|layer2|layer3|layer4)")
    has_depth = j(r"(^|\.)(depth|disp|convs\.)")
    has_pose = j(r"(^|\.)(pose|axisangle|translation|pose_conv|convs\.pose)")
    if has_enc and has_depth and has_pose:
        return "PPGeo_Stage1_Bundle(Encoder+Depth+Pose)"
    if j(r"(^|\.)(layer1|layer2|layer3|layer4)\.") and j(r"(^|\.)(fc|classifier)\."):
        return "ResNet_Classifier_or_PolicyHead"
    if j(r"(^|\.)(layer1|layer2|layer3|layer4)\."):
        return "ResNet_Encoder_like"
    return "Unknown_or_Custom"

def summarize(path, allow_pickle=False, print_keys=False):
    raw, mode = try_load(path, allow_pickle=allow_pickle)
    out = {"path": path, "load_mode": mode, "meta_keys": [], "guess": None,
           "num_params": None, "top_prefix_counts": [], "sample_params": [], "notes": []}

    if isinstance(raw, dict) and "__load_error__" in raw:
        out["notes"].append(raw["__load_error__"])
        return out

    sd, meta = extract_state_dict(raw)
    if sd is None:
        # show top-level keys to help debugging (e.g., Lightning checkpoint)
        if isinstance(raw, dict):
            out["notes"].append(f"Top-level keys: {list(raw.keys())[:20]}")
        else:
            out["notes"].append("No state_dict found; object type: " + type(raw).__name__)
        return out

    out["meta_keys"] = list(meta.keys())
    # Count params
    total = 0
    for v in sd.values():
        if torch.is_tensor(v):
            total += v.numel()
    out["num_params"] = int(total)
    out["guess"] = guess_arch(sd, meta)
    out["top_prefix_counts"] = top_prefixes(list(sd.keys()))
    # sample shapes
    i = 0
    for k, v in sd.items():
        if torch.is_tensor(v):
            out["sample_params"].append({"key": k, "shape": list(v.shape), "numel": int(v.numel())})
            i += 1
            if i >= 12: break
    if print_keys:
        out["notes"].append("ALL_KEYS_START")
        out["notes"] += list(sd.keys())
        out["notes"].append("ALL_KEYS_END")
    # Special: Monodepth2 encoder often stores training resolution in meta (height/width) used by test_simple.py :contentReference[oaicite:5]{index=5}
    if "height" in meta and "width" in meta:
        out["training_resolution"] = [int(meta["height"]), int(meta["width"])]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--allow-pickle", action="store_true",
                    help="**Only for trusted files**. Falls back to torch.load(..., weights_only=False).")
    ap.add_argument("--print-keys", action="store_true")
    ap.add_argument("--save-json", action="store_true")
    args = ap.parse_args()

    for p in args.checkpoints:
        s = summarize(p, allow_pickle=args.allow_pickle, print_keys=args.print_keys)
        print(json.dumps(s, indent=2))
        if args.save_json:
            with open(p + ".arch.json", "w") as f:
                json.dump(s, f, indent=2)
            print(f"[saved] {p}.arch.json")

if __name__ == "__main__":
    main()
