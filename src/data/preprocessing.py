"""Image preprocessing utilities.

Uses PIL-based transforms to ensure MPS compatibility (avoids upsample issues).
"""

import torch
from torchvision import transforms
from PIL import Image


# Standard ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Preprocessing transform for DINOv2
preprocess_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Uses PIL, works on CPU
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


def preprocess_panels(images):
    """
    Convert RAVEN images to DINOv2 input format.

    Uses PIL/torchvision transforms to avoid MPS upsample issues.

    Args:
        images: numpy array of shape [N, 160, 160] (grayscale uint8)

    Returns:
        torch.Tensor of shape [N, 3, 224, 224] (normalized RGB, on CPU)
    """
    panels = []
    for img in images:
        # Convert to PIL Image and to RGB
        pil_img = Image.fromarray(img).convert('RGB')
        # Apply transforms (resize + normalize)
        tensor = preprocess_transform(pil_img)
        panels.append(tensor)
    return torch.stack(panels)  # [N, 3, 224, 224] on CPU
