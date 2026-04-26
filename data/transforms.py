"""
Image Transforms for Histopathology

This module defines augmentation pipelines for:
1. Training (aggressive augmentation)
2. Validation/Test (minimal preprocessing)

Key principles:
- Only use augmentations that preserve biological meaning
- Normalize with ImageNet stats (for transfer learning)
- Resize to 224x224 (ResNet50 input size)
"""

import torch
from torchvision import transforms
import numpy as np
from PIL import Image


# ImageNet Statistics
# Why these values?
# - We're using ResNet50 pretrained on ImageNet
# - ImageNet was normalized with these mean/std during training
# - We MUST use the same normalization for transfer learning to work

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def random_right_angle_rotation(img):
    """Rotate by a random multiple of 90 degrees without free-angle interpolation."""
    angle = int(torch.randint(0, 4, (1,)).item()) * 90
    return transforms.functional.rotate(img, angle)


# Training Augmentation Pipeline

def get_train_transforms(img_size=224):
    """
    Aggressive augmentation for training.

    Augmentations applied:
    1. RandomResizedCrop - crops a random portion, then resizes
       - Simulates zooming in/out on tissue
       - scale=(0.8, 1.0) means crop between 80-100% of original

    2. RandomHorizontalFlip - mirror image left-right
       - Valid: tissue slides have no inherent orientation

    3. RandomVerticalFlip - mirror image top-bottom
       - Valid: same reason as horizontal

    4. RandomRotation - rotate by 0, 90, 180, or 270 degrees
       - Valid: pathologists rotate slides under microscope
       - We ONLY use 90-degree increments (no arbitrary-angle interpolation artifacts)

    5. ColorJitter - slight variations in brightness/contrast/saturation
       - Simulates staining variability between labs
       - Conservative values to avoid unrealistic colors

    6. ToTensor - converts PIL Image to torch.Tensor
       - Scales [0, 255] -> [0.0, 1.0]

    7. Normalize - standardize using ImageNet stats
       - Required for transfer learning

    Args:
        img_size: target size (default 224 for ResNet50)

    Returns:
        transforms.Compose pipeline
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=img_size,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Lambda(random_right_angle_rotation),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# Validation/Test Preprocessing Pipeline

def get_val_transforms(img_size=224):
    """
    Minimal preprocessing for validation and test.

    NO augmentation - we want to evaluate on clean, unmodified images.

    Steps:
    1. Resize to target size (224x224)
    2. CenterCrop (safety - some images might have non-square aspect ratios)
    3. ToTensor
    4. Normalize (same as training)

    Args:
        img_size: target size (default 224)

    Returns:
        transforms.Compose pipeline
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# Utility: Denormalize for Visualization

def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Reverse ImageNet normalization for visualization.

    Normalization formula: x_norm = (x - mean) / std
    Denormalization: x = x_norm * std + mean

    Args:
        tensor: normalized tensor (C, H, W) or (B, C, H, W)
        mean: mean used for normalization
        std: std used for normalization

    Returns:
        denormalized tensor in [0, 1] range
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    mean = mean.to(tensor.device)
    std = std.to(tensor.device)

    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)

    return tensor


def tensor_to_image(tensor):
    """
    Convert a normalized tensor back to PIL Image for visualization.

    Args:
        tensor: (C, H, W) tensor in [-mean/std, (1-mean)/std] range

    Returns:
        PIL Image in [0, 255] range
    """
    tensor = denormalize(tensor)
    img_np = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_np)


# Test Transform Pipeline (if you need it separate from val)

def get_test_transforms(img_size=224):
    """
    Same as validation transforms.

    (Some projects use Test-Time Augmentation here - we'll cover that later.)
    """
    return get_val_transforms(img_size)


# Quick Test

if __name__ == '__main__':
    print("Transform Pipeline Test")
    print("=" * 60)

    dummy_img = Image.new('RGB', (460, 700), color=(128, 64, 200))
    print(f"Input image: {dummy_img.size} (W x H)")

    train_tfm = get_train_transforms()
    train_tensor = train_tfm(dummy_img)
    print(f"Train output: {train_tensor.shape} (C x H x W)")
    print(f"  Min: {train_tensor.min():.3f}, Max: {train_tensor.max():.3f}")
    print(f"  Mean: {train_tensor.mean():.3f}, Std: {train_tensor.std():.3f}")

    val_tfm = get_val_transforms()
    val_tensor = val_tfm(dummy_img)
    print(f"\nVal output: {val_tensor.shape}")
    print(f"  Min: {val_tensor.min():.3f}, Max: {val_tensor.max():.3f}")

    denorm_img = tensor_to_image(val_tensor)
    print(f"\nDenormalized: {denorm_img.size}")

    print("\nTransforms working correctly!")
    print("\nNext step: Create data/dataset.py")
