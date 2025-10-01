from dataset import SegmentationDataset

train_dataset = SegmentationDataset(
    images_dir="dataset/train/images",
    masks_dir="dataset/train/masks",
    transform=None
)

print("Number of train samples:", len(train_dataset))

# Try loading first item
image, mask = train_dataset[0]
print("Image shape:", image.shape)
print("Mask shape:", mask.shape)
