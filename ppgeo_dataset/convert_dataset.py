from PIL import Image
import os
from pathlib import Path

# === CONFIG ===
INPUT_DIR = "/media/sim/data/ivan/ppgeo_dataset/extracted_frames"
OUTPUT_DIR = "/media/sim/data/ivan/ppgeo_dataset/converted_dataset"
CROP_TOP = 10    # PPGeo crops top 10px to match 160 height
CROP_BOTTOM = 170  # i.e., crop to (0, 10, 320, 170)
RESIZE = (320, 160)

def process_image(in_path, out_path):
    img = Image.open(in_path).convert("L")
    img = img.convert("RGB")
    img = img.resize((320, 160))  # match PPGeo expected size exactly
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

def process_all_images(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    count = 0

    for in_path in sorted(input_dir.rglob("*.jpg")):
        rel_path = in_path.relative_to(input_dir)
        out_path = output_dir / rel_path
        process_image(in_path, out_path)
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} images...")

    print(f"\n Done! Total images processed: {count}")

if __name__ == "__main__":
    process_all_images(INPUT_DIR, OUTPUT_DIR)

