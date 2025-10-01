import matplotlib.pyplot as plt
from dataset import SegmentationDataset, get_train_transform

# Change paths according to your dataset
train_images = "denoised_dataset/train/images"
train_masks = "denoised_dataset/train/masks"

# Create dataset object
dataset = SegmentationDataset(
    images_dir=train_images,
    masks_dir=train_masks,
    transform=get_train_transform()
)

print(f"Total samples: {len(dataset)}")

# Show first 2 samples
for i in range(2):
    image, mask = dataset[i]

    # Convert image back to HWC for display
    img = image.transpose(1, 2, 0)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(img)
    ax[0].set_title("Image")
    ax[0].axis("off")

    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Mask")
    ax[1].axis("off")

    plt.show()
