import os
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Paths
DATASET_PATH = r"C:\Users\aksha\OneDrive\Desktop\SegmentationProject\denoised_dataset"
PREVIEW_PATH = os.path.join(DATASET_PATH, "previews")
os.makedirs(PREVIEW_PATH, exist_ok=True)

def show_random_samples(split="train", num_samples=3):
    img_dir = os.path.join(DATASET_PATH, split, "images")
    mask_dir = os.path.join(DATASET_PATH, split, "masks")

    img_files = os.listdir(img_dir)
    mask_files = os.listdir(mask_dir)

    if not img_files or not mask_files:
        print(f"No files found in {img_dir} or {mask_dir}")
        return

    samples = random.sample(img_files, min(num_samples, len(img_files)))

    for i, img_file in enumerate(samples):
        img_path = os.path.join(img_dir, img_file)
        mask_path = os.path.join(mask_dir, img_file.replace(".jpg", ".png"))

        if not os.path.exists(mask_path):
            print(f"Mask not found for {img_file}")
            continue

        # Load image and mask
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # Plot
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.imshow(mask, cmap="gray")
        plt.title("Mask")
        plt.axis("off")

        # Save preview
        save_path = os.path.join(PREVIEW_PATH, f"preview_{i}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Saved {save_path}")

if __name__ == "__main__":
    show_random_samples("train", num_samples=2)
