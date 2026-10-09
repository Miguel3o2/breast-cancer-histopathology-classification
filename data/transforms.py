
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def random_right_angle_rotation(img):
    angle = int(torch.randint(0, 4, (1,)).item()) * 90
    return transforms.functional.rotate(img, angle)


def get_train_transforms(img_size=224):
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



def get_val_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    mean = mean.to(tensor.device)
    std = std.to(tensor.device)

    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)

    return tensor


def tensor_to_image(tensor):
    tensor = denormalize(tensor)
    img_np = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_np)



def get_test_transforms(img_size=224):
    return get_val_transforms(img_size)


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
