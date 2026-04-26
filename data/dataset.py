"""
PyTorch Dataset for BreakHis Breast Cancer Histopathology

This module provides:
1. BreakHisDataset — loads images from CSV with transforms
2. get_dataloaders() — factory function for train/val/test loaders
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from data.transforms import get_train_transforms, get_val_transforms


class BreakHisDataset(Dataset):
    """
    PyTorch Dataset for BreakHis histopathology images.
    
    Loads images lazily (one at a time) to save memory.
    Applies transforms on-the-fly during __getitem__.
    
    Args:
        csv_path: path to train.csv/val.csv/test.csv
        transform: torchvision.transforms pipeline (or None)
        
    Returns:
        (image_tensor, label_tensor) where:
        - image_tensor: (3, 224, 224) float32
        - label_tensor: scalar int (0=benign, 1=malignant)
    """
    
    def __init__(self, csv_path, transform=None):
        """
        Initialize dataset from CSV.
        
        CSV format:
            image_path, label, magnification, subtype, filename
            
        We only need image_path and label for basic classification.
        (magnification and subtype will be used later for multi-task learning)
        """
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        
        # Map string labels to integers
        # 0 = benign, 1 = malignant
        self.label_map = {'benign': 0, 'malignant': 1}
        
        # Verify all image paths exist
        self._verify_paths()
        
        print(f"Loaded BreakHisDataset from {csv_path}")
        print(f"  Total images: {len(self.df)}")
        print(f"  Benign: {(self.df['label'] == 'benign').sum()}")
        print(f"  Malignant: {(self.df['label'] == 'malignant').sum()}")
    
    def _verify_paths(self):
        """
        Check that all image paths in the CSV actually exist.
        Removes missing images from the dataset.
        """
        valid_mask = self.df['image_path'].apply(os.path.exists)
        missing_count = (~valid_mask).sum()
        
        if missing_count > 0:
            print(f"WARNING: {missing_count} images not found, removing from dataset")
            self.df = self.df[valid_mask].reset_index(drop=True)
    
    def __len__(self):
        """Return number of samples."""
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Load and return a single sample.
        
        Args:
            idx: index in [0, len(self)-1]
            
        Returns:
            (image, label) tuple where:
            - image: (3, 224, 224) tensor
            - label: scalar int (0 or 1)
        """
        # Get metadata for this sample
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label = self.label_map[row['label']]
        
        try:
            # Load image as PIL Image
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms (if provided)
            if self.transform is not None:
                image = self.transform(image)
            else:
                # If no transform, convert to tensor manually
                import torchvision.transforms as T
                image = T.ToTensor()(image)
            
            # Convert label to tensor
            label = torch.tensor(label, dtype=torch.long)
            
            return image, label
            
        except Exception as e:
            print(f"ERROR loading {img_path}: {e}")
            # Return a blank image rather than crashing
            # This prevents one corrupted image from breaking training
            blank_image = torch.zeros(3, 224, 224)
            return blank_image, torch.tensor(label, dtype=torch.long)
    
    def get_label_counts(self):
        """
        Return class distribution as a dict.
        Useful for computing class weights.
        """
        return self.df['label'].value_counts().to_dict()
    
    def get_class_weights(self):
        """
        Compute class weights for imbalanced data.
        
        Formula: weight_i = total_samples / (num_classes * count_i)
        
        Returns:
            torch.Tensor of shape (num_classes,)
        """
        label_counts = self.df['label'].value_counts()
        total = len(self.df)
        num_classes = len(label_counts)
        
        weights = {}
        for label, count in label_counts.items():
            weights[self.label_map[label]] = total / (num_classes * count)
        
        # Convert to tensor in order [benign_weight, malignant_weight]
        weight_tensor = torch.tensor([weights[0], weights[1]], dtype=torch.float32)
        
        return weight_tensor


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader Factory
# ──────────────────────────────────────────────────────────────────────────────

def get_dataloaders(batch_size=32, num_workers=4, img_size=224, 
                    data_dir='./data/processed'):
    """
    Create train/val/test DataLoaders with appropriate transforms.
    
    Args:
        batch_size: samples per batch (default 32)
        num_workers: parallel workers for data loading (default 4)
                     Set to 0 on Windows if you get multiprocessing errors
        img_size: image size for resizing (default 224)
        data_dir: directory containing train.csv/val.csv/test.csv
        
    Returns:
        (train_loader, val_loader, test_loader, class_weights)
        
    Example:
        >>> train_dl, val_dl, test_dl, weights = get_dataloaders(batch_size=16)
        >>> images, labels = next(iter(train_dl))
        >>> print(images.shape)   # torch.Size([16, 3, 224, 224])
        >>> print(labels.shape)   # torch.Size([16])
    """
    # Paths to split CSVs
    train_csv = os.path.join(data_dir, 'train.csv')
    val_csv   = os.path.join(data_dir, 'val.csv')
    test_csv  = os.path.join(data_dir, 'test.csv')
    
    # Verify CSVs exist
    for csv_path in [train_csv, val_csv, test_csv]:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"{csv_path} not found. Run data/preprocess.py first."
            )
    
    # Create transforms
    train_tfm = get_train_transforms(img_size)
    val_tfm   = get_val_transforms(img_size)
    
    # Create datasets
    train_dataset = BreakHisDataset(train_csv, transform=train_tfm)
    val_dataset   = BreakHisDataset(val_csv,   transform=val_tfm)
    test_dataset  = BreakHisDataset(test_csv,  transform=val_tfm)
    
    # Get class weights from training set
    class_weights = train_dataset.get_class_weights()
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,              # Shuffle training data
        num_workers=num_workers,
        pin_memory=True,           # Faster CPU->GPU transfer
        drop_last=True             # Drop incomplete final batch (for BatchNorm stability)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,             # No shuffling for validation
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print("\n" + "="*60)
    print("DATALOADERS CREATED")
    print("="*60)
    print(f"Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} images, {len(val_loader)} batches")
    print(f"Test:  {len(test_dataset)} images, {len(test_loader)} batches")
    print(f"\nClass weights: {class_weights}")
    print("="*60)
    
    return train_loader, val_loader, test_loader, class_weights


# ──────────────────────────────────────────────────────────────────────────────
# Quick Test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Testing BreakHisDataset")
    print("="*60)
    
    # Test dataset loading
    try:
        train_csv = './data/processed/train.csv'
        if not os.path.exists(train_csv):
            print(f"ERROR: {train_csv} not found")
            print("Run data/preprocess.py first")
            exit(1)
        
        # Create dataset
        dataset = BreakHisDataset(train_csv, transform=get_train_transforms())
        
        # Load one sample
        img, label = dataset[0]
        print(f"\nSample 0:")
        print(f"  Image shape: {img.shape}")
        print(f"  Label: {label.item()} ({'benign' if label == 0 else 'malignant'})")
        print(f"  Image range: [{img.min():.3f}, {img.max():.3f}]")
        
        # Test DataLoader
        loader = DataLoader(dataset, batch_size=4, num_workers=0)
        batch_imgs, batch_labels = next(iter(loader))
        print(f"\nBatch:")
        print(f"  Images: {batch_imgs.shape}")
        print(f"  Labels: {batch_labels}")
        
        print("\n✓ Dataset working correctly!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
