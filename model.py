# model.py
import segmentation_models_pytorch as smp

def get_unet_model():
    """
    Returns the exact same U-Net architecture used during training.
    (encoder = resnet18, imagenet pretraining, 3 input channels, 1 output class)
    """
    model = smp.Unet(
        encoder_name="resnet18",     # ✅ must match training
        encoder_weights="imagenet",  # ✅ same as training
        in_channels=3,
        classes=1
    )
    return model

