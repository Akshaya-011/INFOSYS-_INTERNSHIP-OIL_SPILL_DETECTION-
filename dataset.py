import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class SegmentationDataset(Dataset):
    """
    Robust dataset for binary segmentation.
    - Reads images (RGB) and masks (grayscale).
    - Binarizes masks to 0/1.
    - Works with albumentations (preferred) but also safe if transform returns numpy.
    - Returns: image Tensor [C,H,W] (float32), mask Tensor [1,H,W] (float32, values 0 or 1).
    """
    def __init__(self, images_dir, masks_dir, transform=None, img_exts=(".jpg",".jpeg",".png")):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.images = sorted([
            os.path.join(images_dir, f) for f in os.listdir(images_dir)
            if f.lower().endswith(img_exts)
        ])
        self.masks = sorted([
            os.path.join(masks_dir, f) for f in os.listdir(masks_dir)
            if f.lower().endswith(img_exts)
        ])

        assert len(self.images) == len(self.masks), f"Images ({len(self.images)}) and masks ({len(self.masks)}) count mismatch!"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        # Load
        image = np.array(Image.open(img_path).convert("RGB"))   # H,W,3 uint8
        mask  = np.array(Image.open(mask_path).convert("L"))    # H,W uint8

        # Binarize mask to {0,1}
        mask = (mask > 127).astype(np.uint8)

        # Apply transform (albumentations style: transform(image=..., mask=...))
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"]

        # --- Normalize types and shapes (robust to transform output types) ---
        # Image -> torch.FloatTensor [C,H,W], values ~[0,1] if ToTensorV2 used, else convert.
        if isinstance(image, np.ndarray):
            # Convert HWC uint8 [0..255] -> float32 [0..1]
            image = torch.from_numpy(image.transpose(2,0,1)).float().div(255.0)
        elif isinstance(image, torch.Tensor):
            # If albumentations ToTensorV2 used, image likely is already [C,H,W] float
            if image.dim() == 3 and image.shape[0] in (1,3,4):
                image = image.float()
            else:
                # If shape is HWC, convert
                if image.dim() == 3:
                    image = image.permute(2,0,1).float()

        # Mask -> torch.FloatTensor [1,H,W] with 0/1
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()     # long or uint8
        elif isinstance(mask, torch.Tensor):
            mask = mask.long()

        # ensure shape [1,H,W]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        elif mask.dim() == 3 and mask.shape[0] != 1:
            # If mask is [H,W,1] or [C,H,W] fallback: try squeeze last dim
            if mask.shape[-1] == 1:
                mask = mask.squeeze(-1)
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)
            else:
                # take first channel
                mask = mask[0].unsqueeze(0)

        # Convert to float 0/1
        mask = (mask > 0).float()

        return image.contiguous(), mask.contiguous()
