import cv2
import matplotlib.pyplot as plt
import os

# Paths
base_path = r"C:\Users\aksha\Downloads\SegmentationProject\dataset\test"
images_path = os.path.join(base_path, "images")
masks_path = os.path.join(base_path, "masks")

# Get first 3 images automatically
all_images = sorted(os.listdir(images_path))[:3]

# Folder to save combined comparison
save_folder = os.path.join(base_path, "comparison")
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder, "three_images_comparison.png")

# Create figure
plt.figure(figsize=(12, 4 * len(all_images)))  # adjust height for rows

for i, img_file in enumerate(all_images):
    # Load original and mask
    original = cv2.imread(os.path.join(images_path, img_file))
    mask_name = img_file.rsplit('.', 1)[0] + ".png"
    mask_gt = cv2.imread(os.path.join(masks_path, mask_name), cv2.IMREAD_GRAYSCALE)
    mask_pred = mask_gt.copy()  # placeholder

    if original is None or mask_gt is None:
        print(f"❌ Could not load {img_file} or its mask, skipping...")
        continue

    # Plot original, mask_gt, mask_pred in a row
    plt.subplot(len(all_images), 3, i*3 + 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title(f"Original {i+1}")
    plt.axis('off')

    plt.subplot(len(all_images), 3, i*3 + 2)
    plt.imshow(mask_gt, cmap='jet')
    plt.title(f"Ground Truth {i+1}")
    plt.axis('off')

    plt.subplot(len(all_images), 3, i*3 + 3)
    plt.imshow(mask_pred, cmap='jet')
    plt.title(f"Prediction {i+1}")
    plt.axis('off')

plt.tight_layout()

# Save figure directly, no preview
plt.savefig(save_path)
plt.close()

print(f"✅ Combined comparison saved at: {save_path}")
