import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -----------------------------
# Paths
# -----------------------------
GT_PATH = "predictions/gt_Oil (4).png"
PRED_PATH = "predictions/pred_Oil (4).png"
REPORT_DIR = "report_plots"

os.makedirs(REPORT_DIR, exist_ok=True)

# -----------------------------
# Load images
# -----------------------------
gt = cv2.imread(GT_PATH, cv2.IMREAD_GRAYSCALE)
pred = cv2.imread(PRED_PATH, cv2.IMREAD_GRAYSCALE)

if gt is None or pred is None:
    raise FileNotFoundError("❌ Check GT/PRED paths – files not found or unreadable.")

print("✅ Loaded images:", gt.shape, pred.shape)

# -----------------------------
# Binarize masks
# -----------------------------
gt_bin = (gt > 127).astype(np.uint8)
pred_bin = (pred > 127).astype(np.uint8)

# -----------------------------
# Metrics
# -----------------------------
TP = np.sum((gt_bin == 1) & (pred_bin == 1))
TN = np.sum((gt_bin == 0) & (pred_bin == 0))
FP = np.sum((gt_bin == 0) & (pred_bin == 1))
FN = np.sum((gt_bin == 1) & (pred_bin == 0))

iou = TP / (TP + FP + FN + 1e-8)
dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)
precision = TP / (TP + FP + 1e-8)
recall = TP / (TP + FN + 1e-8)

metrics = {
    "IoU": iou,
    "Dice": dice,
    "Precision": precision,
    "Recall": recall
}

print("📊 Metrics:", metrics)

# -----------------------------
# Save Metrics Bar Plot
# -----------------------------
plt.figure(figsize=(6,4))
plt.bar(metrics.keys(), metrics.values(), color="skyblue")
plt.ylim(0,1)
plt.ylabel("Score")
plt.title("Segmentation Metrics")
plt.savefig(os.path.join(REPORT_DIR, "metrics.png"))
plt.close()

# -----------------------------
# Confusion Matrix
# -----------------------------
y_true = gt_bin.flatten()
y_pred = pred_bin.flatten()
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Background","Oil Spill"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"))
plt.close()

# -----------------------------
# Overlay visualization
# -----------------------------
overlay = cv2.addWeighted(gt_bin*255, 0.5, pred_bin*255, 0.5, 0)

fig, axs = plt.subplots(1,3, figsize=(12,4))
axs[0].imshow(gt_bin, cmap="gray")
axs[0].set_title("Ground Truth")
axs[1].imshow(pred_bin, cmap="gray")
axs[1].set_title("Prediction")
axs[2].imshow(overlay, cmap="jet")
axs[2].set_title("Overlay (GT vs Pred)")

for ax in axs:
    ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "comparison.png"))
plt.close()

print(f"\n📂 Report saved in: {REPORT_DIR}")
