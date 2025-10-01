import os
import cv2
import numpy as np
from tqdm import tqdm

# Path to dataset folder
DATASET_PATH = "dataset"   # because it's inside SegmentationProject

# Path to save preprocessed images
OUTPUT_PATH = "processed_dataset"

# Create output folder if not exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Resize settings
IMG_SIZE = (128, 128)

def preprocess_images():
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in tqdm(files, desc=f"Processing in {root}"):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(root, file)

                # read image
                img = cv2.imread(file_path, cv2.IMREAD_COLOR)

                # resize
                img_resized = cv2.resize(img, IMG_SIZE)

                # normalize (0–1)
                img_normalized = img_resized / 255.0

                # save as .npy file
                save_path = os.path.join(OUTPUT_PATH, file.split('.')[0] + ".npy")
                np.save(save_path, img_normalized)

    print("✅ Preprocessing complete. Files saved in", OUTPUT_PATH)


if __name__ == "__main__":
    preprocess_images()
