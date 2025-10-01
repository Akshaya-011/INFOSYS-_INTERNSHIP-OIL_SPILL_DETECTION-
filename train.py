import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from dataset import SegmentationDataset

# ------------------------
# CONFIG - edit these only if needed
# ------------------------
POSSIBLE_TRAIN_IMG_DIRS = ["denoised_dataset/train/images", "dataset/train/images"]
POSSIBLE_TRAIN_MASK_DIRS = ["denoised_dataset/train/masks", "dataset/train/masks"]
POSSIBLE_VAL_IMG_DIRS   = ["denoised_dataset/val/images", "dataset/val/images"]
POSSIBLE_VAL_MASK_DIRS  = ["denoised_dataset/val/masks", "dataset/val/masks"]

IMG_SIZE = 128          # smaller -> faster
BATCH_SIZE = 4          # CPU: 2 or 4; GPU: increase as fits
NUM_WORKERS = 0         # 0 on Windows is safest
EPOCHS = 10
LEARNING_RATE = 1e-4
ENCODER = "resnet18"    # good tradeoff (use "mobilenet_v2" for even faster)
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------
# helper: choose existing dir from list
# ------------------------
def choose_first_exist(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

train_img_dir = choose_first_exist(POSSIBLE_TRAIN_IMG_DIRS)
train_mask_dir = choose_first_exist(POSSIBLE_TRAIN_MASK_DIRS)
val_img_dir   = choose_first_exist(POSSIBLE_VAL_IMG_DIRS)
val_mask_dir  = choose_first_exist(POSSIBLE_VAL_MASK_DIRS)

if not (train_img_dir and train_mask_dir):
    raise FileNotFoundError("Training image/mask folders not found. Update POSSIBLE_*DIRS in train.py")

if not (val_img_dir and val_mask_dir):
    print("Warning: Validation folders not found. Using train set as validation for quick test.")
    val_img_dir, val_mask_dir = train_img_dir, train_mask_dir

print("Using:")
print(" TRAIN IMAGES:", train_img_dir)
print(" TRAIN MASKS: ", train_mask_dir)
print(" VAL IMAGES:  ", val_img_dir)
print(" VAL MASKS:   ", val_mask_dir)

# ------------------------
# AUGMENTATIONS
# ------------------------
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.3),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

# ------------------------
# DATA
# ------------------------
train_dataset = SegmentationDataset(train_img_dir, train_mask_dir, transform=train_transform)
val_dataset   = SegmentationDataset(val_img_dir, val_mask_dir, transform=val_transform)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("CUDA available - training on GPU.")
else:
    print("No CUDA - training on CPU (slower).")

# adjust batch size automatically for CPU
if not torch.cuda.is_available() and BATCH_SIZE > 4:
    BATCH_SIZE = 4

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
print(f"Batch size: {BATCH_SIZE}, Image size: {IMG_SIZE}x{IMG_SIZE}")

# ------------------------
# MODEL, LOSS, OPTIM
# ------------------------
model = smp.Unet(encoder_name=ENCODER, encoder_weights="imagenet", in_channels=3, classes=1)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ------------------------
# METRICS (robust batching)
# ------------------------
def batch_metrics(outputs, masks, threshold=0.5):
    """
    outputs: raw logits from model, shape [B,1,H,W]
    masks: binary floats 0/1, shape [B,1,H,W]
    returns: (intersection_sum, union_sum, correct_pixels, total_pixels)
    """
    probs = torch.sigmoid(outputs)
    preds = (probs > threshold).float()

    intersection = (preds * masks).sum().item()
    union = (preds + masks - preds * masks).sum().item()
    correct = (preds == masks).sum().item()
    total = preds.numel()
    return intersection, union, correct, total

# ------------------------
# TRAINING LOOP
# ------------------------
best_val_iou = -1.0

try:
    for epoch in range(1, EPOCHS + 1):
        # ------- train -------
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)      # [B,3,H,W]
            masks  = masks.to(device)       # [B,1,H,W], float 0/1 ensured in dataset

            optimizer.zero_grad()
            outputs = model(images)         # logits [B,1,H,W]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / max(1, len(train_loader))

        # ------- validate -------
        model.eval()
        val_loss = 0.0
        total_inter, total_union, total_correct, total_pixels = 0.0, 0.0, 0, 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks  = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                inter, union, correct, total = batch_metrics(outputs, masks)
                total_inter += inter
                total_union += union
                total_correct += correct
                total_pixels += total

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_iou = (total_inter / total_union) if total_union > 0 else 1.0
        val_acc = (total_correct / total_pixels) if total_pixels > 0 else 0.0

        print(f"Epoch [{epoch}/{EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {val_iou:.4f} | Val Acc: {val_acc:.4f}")

        # save best
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
            print(f" -> Saved best_model.pth (Val IoU {best_val_iou:.4f})")

except KeyboardInterrupt:
    print("Training interrupted by user, saving last model...")
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "last_model.pth"))
    print("Saved last_model.pth. Exiting.")
