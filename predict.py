# predict.py
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import get_unet_model

# ------------------------
# CONFIG
# ------------------------
CHECKPOINT_CANDIDATES = [
    "checkpoints/best_model.pth",
    "best_model.pth",
    "checkpoints/last_model.pth",
    "last_model.pth"
]
TEST_IMAGE_PATH = r"C:\Users\aksha\OneDrive\Desktop\SegmentationProject\dataset\test\images\Oil (4).jpg"
IMG_SIZE = 128   # must match training
THRESHOLD = 0.5
OUTPUT_DIR = "predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------
# DEVICE
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ------------------------
# LOAD MODEL
# ------------------------
checkpoint_path = None
for p in CHECKPOINT_CANDIDATES:
    if os.path.exists(p):
        checkpoint_path = p
        break
if checkpoint_path is None:
    raise FileNotFoundError("No checkpoint found!")

print("Using checkpoint:", checkpoint_path)
model = get_unet_model().to(device)

ckpt = torch.load(checkpoint_path, map_location=device)
if isinstance(ckpt, dict):
    if "model_state" in ckpt:
        state = ckpt["model_state"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
else:
    state = ckpt
model.load_state_dict(state)
model.eval()
print("Model loaded.")

# ------------------------
# HELPERS
# ------------------------
def preprocess_image(img_path, img_size=IMG_SIZE):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found at {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    img_resized = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img_norm = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_norm - mean) / std

    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float()
    return img_rgb, img_tensor.to(device), (orig_w, orig_h)

def predict_mask(img_tensor, threshold=THRESHOLD):
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits.cpu())
        pred = (probs > threshold).float().numpy()[0, 0]
    return pred

def resize_mask_to_orig(pred_mask, orig_size):
    orig_w, orig_h = orig_size
    pred_uint8 = (pred_mask * 255).astype(np.uint8)
    resized = cv2.resize(pred_uint8, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return resized

def find_ground_truth_mask(image_path):
    images_dir, fname = os.path.split(image_path)
    parts = images_dir.split(os.sep)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "masks"
        masks_dir = os.sep.join(parts)
    else:
        masks_dir = os.path.join(os.path.dirname(images_dir), "masks")

    base, _ = os.path.splitext(fname)
    candidates = [os.path.join(masks_dir, base + ext) for ext in [".png", ".jpg", ".jpeg"]]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def save_outputs(orig_rgb, pred_resized, gt_img, out_base_name):
    # save predicted mask
    pred_path = os.path.join(OUTPUT_DIR, f"pred_{out_base_name}.png")
    cv2.imwrite(pred_path, pred_resized)

    # save overlay
    overlay = orig_rgb.copy()
    white_mask = np.ones_like(overlay, dtype=np.uint8) * 255
    alpha = 0.4
    mask_bool = pred_resized.astype(bool)
    overlay[mask_bool] = cv2.addWeighted(orig_rgb[mask_bool], 1.0, white_mask[mask_bool], alpha, 0)

    overlay_path = os.path.join(OUTPUT_DIR, f"overlay_{out_base_name}.png")
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # save GT mask if available
    gt_path = None
    if gt_img is not None:
        gt_path = os.path.join(OUTPUT_DIR, f"gt_{out_base_name}.png")
        cv2.imwrite(gt_path, gt_img)

    return pred_path, overlay_path, gt_path

# ------------------------
# RUN
# ------------------------
print("Test image:", TEST_IMAGE_PATH)
orig_rgb, tensor, orig_size = preprocess_image(TEST_IMAGE_PATH, IMG_SIZE)
pred_mask = predict_mask(tensor, THRESHOLD)
pred_resized = resize_mask_to_orig(pred_mask, orig_size)

# ground truth
gt_img = None
gt_path = find_ground_truth_mask(TEST_IMAGE_PATH)
if gt_path:
    gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt_img is not None:
        gt_img = cv2.resize(gt_img, (orig_size[0], orig_size[1]), interpolation=cv2.INTER_NEAREST)

base_name = os.path.splitext(os.path.basename(TEST_IMAGE_PATH))[0]
pred_path, overlay_path, saved_gt = save_outputs(orig_rgb, pred_resized, gt_img, base_name)

print("✅ Saved predicted mask to:", pred_path)
print("✅ Saved overlay to:", overlay_path)
if saved_gt:
    print("✅ Saved ground truth mask to:", saved_gt)
else:
    print("No ground truth mask found.")

# ------------------------
# VISUALIZE
# ------------------------
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(orig_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
if gt_img is not None:
    plt.imshow(gt_img, cmap="gray")
    plt.title("Ground Truth Mask")
else:
    plt.text(0.5, 0.5, "No GT mask found", ha="center", va="center")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(pred_resized, cmap="gray")
plt.title("Predicted Mask")
plt.axis("off")

plt.tight_layout()
plt.show()
