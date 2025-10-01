import os
import numpy as np
import cv2
from glob import glob

# Input dataset folders (update if needed)
IMAGE_DIR = r"C:\Users\aksha\OneDrive\Desktop\SegmentationProject\dataset\images"
MASK_DIR = r"C:\Users\aksha\OneDrive\Desktop\SegmentationProject\dataset\masks"

# Output dataset folders
OUT_IMAGE_DIR = r"C:\Users\aksha\OneDrive\Desktop\SegmentationProject\denoised_dataset\train\images"
OUT_MASK_DIR = r"C:\Users\aksha\OneDrive\Desktop\SegmentationProject\denoised_dataset\train\masks"

# Make sure output directories exist
os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

def add_speckle_noise(image):
    row, col, ch = image.shape
    gauss = np.random.randn(row, col, ch)
    noisy = image + image * gauss * 0.1  # noise strength
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def preprocess_and_save():
    image_files = sorted(glob(os.path.join(IMAGE_DIR, "*.png"))) + sorted(glob(os.path.join(IMAGE_DIR, "*.jpg")))
    mask_files = sorted(glob(os.path.join(MASK_DIR, "*.png"))) + sorted(glob(os.path.join(MASK_DIR, "*.jpg")))

    print(f"Found {len(image_files)} images and {len(mask_files)} masks")

    for idx, (img_path, mask_path) in enumerate(zip(image_files, mask_files)):
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"⚠️ Skipping {img_path} or {mask_path} (couldn’t load)")
            continue

        # Resize
        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        # Add noise to image
        img = add_speckle_noise(img)

        # Normalize image to 0-1
        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.uint8)

        # Save as .npy
        img_out = os.path.join(OUT_IMAGE_DIR, f"image_{idx+1}.npy")
        mask_out = os.path.join(OUT_MASK_DIR, f"mask_{idx+1}.npy")

        np.save(img_out, img)
        np.save(mask_out, mask)

        print(f"✅ Saved {img_out}, {mask_out}")

if __name__ == "__main__":
    preprocess_and_save()
