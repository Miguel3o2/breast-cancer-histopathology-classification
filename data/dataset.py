
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from data.transforms import get_train_transforms, get_val_transforms


class BreakHisDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        
        self.label_map = {'benign': 0, 'malignant': 1}
        
        self._verify_paths()
        
        print(f"Loaded BreakHisDataset from {csv_path}")
        print(f"  Total images: {len(self.df)}")
        print(f"  Benign: {(self.df['label'] == 'benign').sum()}")
        print(f"  Malignant: {(self.df['label'] == 'malignant').sum()}")
    
    def _verify_paths(self):
        valid_mask = self.df['image_path'].apply(os.path.exists)
        missing_count = (~valid_mask).sum()
        
        if missing_count > 0:
            print(f"WARNING: {missing_count} images not found, removing from dataset")
            self.df = self.df[valid_mask].reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label = self.label_map[row['label']]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform is not None:
                image = self.transform(image)
            else:
                import torchvision.transforms as T
                image = T.ToTensor()(image)
            
            label = torch.tensor(label, dtype=torch.long)
            
            return image, label
            
        except Exception as e:
            print(f"ERROR loading {img_path}: {e}")
            blank_image = torch.zeros(3, 224, 224)
            return blank_image, torch.tensor(label, dtype=torch.long)
    
    def get_label_counts(self):
        return self.df['label'].value_counts().to_dict()
    
    def get_class_weights(self):
        label_counts = self.df['label'].value_counts()
        total = len(self.df)
        num_classes = len(label_counts)
        
        weights = {}
        for label, count in label_counts.items():
            weights[self.label_map[label]] = total / (num_classes * count)
        
        weight_tensor = torch.tensor([weights[0], weights[1]], dtype=torch.float32)
        
        return weight_tensor


def get_dataloaders(batch_size=32, num_workers=4, img_size=224, 
                    data_dir='./data/processed'):
    train_csv = os.path.join(data_dir, 'train.csv')
    val_csv   = os.path.join(data_dir, 'val.csv')
    test_csv  = os.path.join(data_dir, 'test.csv')
    
    for csv_path in [train_csv, val_csv, test_csv]:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"{csv_path} not found. Run data/preprocess.py first."
            )
    
    train_tfm = get_train_transforms(img_size)
    val_tfm   = get_val_transforms(img_size)
    
    train_dataset = BreakHisDataset(train_csv, transform=train_tfm)
    val_dataset   = BreakHisDataset(val_csv,   transform=val_tfm)
    test_dataset  = BreakHisDataset(test_csv,  transform=val_tfm)
    
    class_weights = train_dataset.get_class_weights()
    
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


if __name__ == '__main__':
    print("Testing BreakHisDataset")
    print("="*60)
    
    try:
        train_csv = './data/processed/train.csv'
        if not os.path.exists(train_csv):
            print(f"ERROR: {train_csv} not found")
            print("Run data/preprocess.py first")
            exit(1)
        
        dataset = BreakHisDataset(train_csv, transform=get_train_transforms())
        
        img, label = dataset[0]
        print(f"\nSample 0:")
        print(f"  Image shape: {img.shape}")
        print(f"  Label: {label.item()} ({'benign' if label == 0 else 'malignant'})")
        print(f"  Image range: [{img.min():.3f}, {img.max():.3f}]")
        
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
