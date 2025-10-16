import matplotlib.pyplot as plt
import os

# Example metric data
epochs = list(range(1, 11))
dice_scores = [0.60, 0.65, 0.70, 0.73, 0.75, 0.78, 0.80, 0.82, 0.83, 0.85]
iou_scores  = [0.50, 0.55, 0.60, 0.63, 0.65, 0.68, 0.70, 0.72, 0.73, 0.75]
accuracy    = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.83, 0.84, 0.85]
loss        = [0.9, 0.8, 0.7, 0.65, 0.6, 0.55, 0.50, 0.48, 0.45, 0.42]

plt.figure(figsize=(8,5))

# Plot Dice, IoU, and Accuracy on left Y-axis
plt.plot(epochs, dice_scores, marker='o', color='b', label='Dice')
plt.plot(epochs, iou_scores, marker='s', color='r', label='IoU')
plt.plot(epochs, accuracy, marker='^', color='g', label='Accuracy')

# Plot Loss on a separate right Y-axis
plt2 = plt.twinx()
plt2.plot(epochs, loss, marker='x', color='purple', linestyle='--', label='Loss')
plt2.set_ylabel("Loss")

# Labels and titles
plt.xlabel("Epoch")
plt.ylabel("Score (Dice / IoU / Accuracy)")
plt.title("Segmentation Metrics over Epochs")
plt.grid(True)

# Combine legends from both axes
lines, labels = plt.gca().get_legend_handles_labels()
lines2, labels2 = plt2.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, loc='lower right')

# Save to your comparison folder
save_dir = r"C:\Users\aksha\Downloads\SegmentationProject\dataset\comparison"
os.makedirs(save_dir, exist_ok=True)  # Create folder if missing
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "all_metrics_curves.png"))
plt.close()

print(f"✅ All metrics curves saved to: {os.path.join(save_dir, 'all_metrics_curves.png')}")
