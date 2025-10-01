# evaluate.py
import os
import cv2
import torch
import numpy as np
from glob import glob
from sklearn.metrics import jaccard_score, f1_score, accuracy_score
from model import get_unet_model  # same as your training file

# -----------------
# CONFIG
# -----------------
CHECKPOINT = "checkpoints/best_model.pth"   # your saved model
TEST_DIR = r"C:\Users\aksha\OneDrive\Desktop\SegmentationProject\dataset\test"
IMG_SIZE = 128
THRESHOLD = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -----------------
# Load Model
# -----------------
model = get_unet_model().to(device)
state = torch.load(CHECKPOINT, map_location=device)
if "model_state" in state:
    state = state["model_state"]
elif "state_dict" in state:
    state = state["state_dict"]
model.load_state_dict(state)
model.eval()
print("Loaded model from", CHECKPOINT)

# -----------------
# Helpers
# -----------------
def preprocess(img_path, img_size=IMG_SIZE):
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]
    img_resized = cv2.resize(img_rgb, (img_size, img_size))

    img_norm = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_norm - mean) / std
    tensor = torch.from_numpy(img_norm.transpose(2,0,1)).unsqueeze(0).float().to(device)

    return tensor, (orig_w, orig_h)

def predict_mask(tensor):
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0,0]
        pred = (probs > THRESHOLD).astype(np.uint8)
    return pred

# -----------------
# Evaluation Loop
# -----------------
image_paths = glob(os.path.join(TEST_DIR, "images", "*.jpg"))  # adjust ext if needed
ious, dices, accs = [], [], []

for img_path in image_paths[:5]:  # just check first 5 images
    base = os.path.splitext(os.path.basename(img_path))[0]
    mask_path = os.path.join(TEST_DIR, "masks", base + ".png")

    if not os.path.exists(mask_path):
        continue

    tensor, orig_size = preprocess(img_path)
    pred = predict_mask(tensor)

    # resize GT to same as pred
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.resize(gt_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    gt_mask = (gt_mask > 127).astype(np.uint8)

    # metrics
    ious.append(jaccard_score(gt_mask.flatten(), pred.flatten()))
    dices.append(f1_score(gt_mask.flatten(), pred.flatten()))
    accs.append(accuracy_score(gt_mask.flatten(), pred.flatten()))

print("Evaluation Results:")
print("Mean IoU:  ", np.mean(ious))
print("Mean Dice: ", np.mean(dices))
print("Accuracy:  ", np.mean(accs))
