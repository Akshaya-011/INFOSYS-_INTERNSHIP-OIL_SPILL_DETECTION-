import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report

# ==========================
# 🔹 Example Data (replace with your real labels)
# ==========================
# true and predicted labels for test images
y_true = ['road', 'road', 'building', 'tree', 'road', 'tree', 'sky', 'sky', 'building', 'tree']
y_pred = ['road', 'building', 'building', 'tree', 'road', 'tree', 'road', 'sky', 'building', 'tree']

classes = sorted(list(set(y_true + y_pred)))  # get all unique classes

# ==========================
# 🔹 Save Folder Setup
# ==========================
save_dir = r"C:\Users\aksha\Downloads\SegmentationProject\dataset\comparison"
os.makedirs(save_dir, exist_ok=True)

# ==========================
# 1️⃣ Confusion Matrix Heatmap
# ==========================
cm = confusion_matrix(y_true, y_pred, labels=classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "1_confusion_matrix.png"))
plt.close()

# ==========================
# 2️⃣ Classification Report Bar Chart
# ==========================
report = classification_report(y_true, y_pred, labels=classes, output_dict=True)
precision = [report[c]['precision'] for c in classes]
recall = [report[c]['recall'] for c in classes]
f1 = [report[c]['f1-score'] for c in classes]

x = np.arange(len(classes))
width = 0.25

plt.figure(figsize=(8,5))
plt.bar(x - width, precision, width, label='Precision')
plt.bar(x, recall, width, label='Recall')
plt.bar(x + width, f1, width, label='F1-Score')
plt.xticks(x, classes)
plt.ylim(0,1)
plt.title("Per-Class Metrics")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "2_classification_metrics_bar.png"))
plt.close()

# ==========================
# 3️⃣ Pie Chart of Overall Accuracy Composition
# ==========================
accuracy = np.trace(cm) / np.sum(cm)
misclassified = 1 - accuracy

plt.figure(figsize=(5,5))
plt.pie(
    [accuracy, misclassified],
    labels=['Correct Predictions', 'Misclassifications'],
    autopct='%1.1f%%',
    colors=['#4CAF50', '#E74C3C'],
    startangle=120
)
plt.title("Prediction Accuracy Composition")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "3_accuracy_pie.png"))
plt.close()

# ==========================
# ✅ Done
# ==========================
print(f"✅ All confusion matrix visualizations saved in:\n{save_dir}")
