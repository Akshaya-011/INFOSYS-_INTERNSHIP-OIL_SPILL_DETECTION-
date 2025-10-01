# full_speckle_denoise.py
import os, cv2
from glob import glob

# Input dataset paths
DATASET_DIR = r"dataset"
OUTPUT_DIR = r"denoised_dataset"

# Subfolders we expect
SUBSETS = ["train", "val", "test"]
CATEGORIES = ["images", "masks"]

# Create output directories
for subset in SUBSETS:
    for cat in CATEGORIES:
        os.makedirs(os.path.join(OUTPUT_DIR, subset, cat), exist_ok=True)

def denoise_and_save(input_path, output_path, is_mask=False):
    img = cv2.imread(input_path)
    if img is None:
        print("⚠️ Skipping unreadable file:", input_path)
        return

    if is_mask:
        # Masks should not be blurred — just copy them
        denoised = img
    else:
        # Median filter for speckle noise reduction
        denoised = cv2.medianBlur(img, 3)   # keep kernel small to avoid artifacts

    cv2.imwrite(output_path, denoised)

# Process all subsets
for subset in SUBSETS:
    for cat in CATEGORIES:
        input_dir = os.path.join(DATASET_DIR, subset, cat)
        output_dir = os.path.join(OUTPUT_DIR, subset, cat)

        files = glob(os.path.join(input_dir, "*.jpg")) + glob(os.path.join(input_dir, "*.png"))
        print(f"{subset}/{cat}: Found {len(files)} files")

        for i, file in enumerate(files, 1):
            filename = os.path.basename(file)
            out_path = os.path.join(output_dir, filename)
            denoise_and_save(file, out_path, is_mask=(cat == "masks"))

        print(f"✅ Done {subset}/{cat}")


