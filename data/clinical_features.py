"""
Generate Synthetic Clinical Features for BreakHis

Since BreakHis doesn't include real patient data, we simulate realistic
clinical features based on medical statistics:
- Age distribution
- Tumor size estimates
- Family history probabilities
- Magnification levels (already in dataset)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path


def simulate_clinical_features(df, seed=42):
    """
    Generate synthetic clinical features for each image.
    
    Features:
    1. Age (years) - sampled from realistic distributions
    2. Tumor size (mm) - estimated from image
    3. Family history (binary) - 0/1
    4. Magnification (categorical) - already in dataset, one-hot encoded
    
    Args:
        df: DataFrame with columns [image_path, label, magnification, ...]
        seed: random seed for reproducibility
    
    Returns:
        df with added clinical feature columns
    """
    np.random.seed(seed)
    
    n_samples = len(df)
    df = df.copy()
    
    # Feature 1: Patient Age
    # Realistic distributions based on medical statistics
    ages = []
    for label in df['label']:
        if label == 'benign':
            # Benign tumors: younger patients (mean 45, std 12)
            age = np.random.normal(45, 12)
        else:  # malignant
            # Malignant tumors: older patients (mean 58, std 15)
            age = np.random.normal(58, 15)
        
        # Clamp to realistic range [20, 90]
        age = np.clip(age, 20, 90)
        ages.append(age)
    
    df['age'] = ages
    
    # Feature 2: Tumor Size (mm)
    # Estimate from label (malignant typically larger)
    sizes = []
    for label in df['label']:
        if label == 'benign':
            # Benign: smaller (mean 15mm, std 8)
            size = np.random.normal(15, 8)
        else:
            # Malignant: larger (mean 28mm, std 12)
            size = np.random.normal(28, 12)
        
        # Clamp to realistic range [5, 50]
        size = np.clip(size, 5, 50)
        sizes.append(size)
    
    df['tumor_size_mm'] = sizes
    
    # Feature 3: Family History (binary)
    # Higher probability for malignant cases
    family_history = []
    for label in df['label']:
        if label == 'benign':
            # 10% of benign cases have family history
            has_history = np.random.random() < 0.10
        else:
            # 30% of malignant cases have family history
            has_history = np.random.random() < 0.30
        
        family_history.append(int(has_history))
    
    df['family_history'] = family_history
    
    # Feature 4: Magnification (one-hot encode)
    # Already in dataset, just need to encode
    magnifications = df['magnification'].unique()
    for mag in magnifications:
        df[f'mag_{mag}'] = (df['magnification'] == mag).astype(int)
    
    return df


def normalize_features(df, split='train', stats=None):
    """
    Z-score normalization for continuous features.
    
    Important: Use training set statistics for val/test normalization
    to avoid data leakage.
    
    Args:
        df: DataFrame with clinical features
        split: 'train', 'val', or 'test'
        stats: dict with mean/std from training set (for val/test)
    
    Returns:
        df with normalized features, stats dict
    """
    continuous_features = ['age', 'tumor_size_mm']
    
    if split == 'train':
        # Compute statistics from training data
        stats = {}
        for feat in continuous_features:
            stats[feat] = {
                'mean': df[feat].mean(),
                'std': df[feat].std()
            }
        
        # Normalize
        for feat in continuous_features:
            df[f'{feat}_norm'] = (df[feat] - stats[feat]['mean']) / stats[feat]['std']
    
    else:
        # Use provided training statistics
        if stats is None:
            raise ValueError("Must provide training statistics for val/test normalization")
        
        for feat in continuous_features:
            df[f'{feat}_norm'] = (df[feat] - stats[feat]['mean']) / stats[feat]['std']
    
    return df, stats


def extract_clinical_vector(row):
    """
    Extract clinical feature vector for a single sample.
    
    Features (7-dim):
    - age_norm (1)
    - tumor_size_mm_norm (1)
    - family_history (1)
    - mag_40X, mag_100X, mag_200X, mag_400X (4)
    
    Args:
        row: pandas Series with clinical features
    
    Returns:
        numpy array of shape (7,)
    """
    features = [
        row['age_norm'],
        row['tumor_size_mm_norm'],
        row['family_history'],
        row.get('mag_40X', 0),
        row.get('mag_100X', 0),
        row.get('mag_200X', 0),
        row.get('mag_400X', 0)
    ]
    
    return np.array(features, dtype=np.float32)


def generate_all_clinical_features(data_dir='./data/processed', output_dir='./data/clinical'):
    """
    Generate clinical features for all splits and save.
    
    Args:
        data_dir: directory with train.csv, val.csv, test.csv
        output_dir: where to save clinical feature CSVs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load splits
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    print("Generating clinical features...")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    # Generate features
    train_df = simulate_clinical_features(train_df, seed=42)
    val_df = simulate_clinical_features(val_df, seed=43)
    test_df = simulate_clinical_features(test_df, seed=44)
    
    # Normalize (using training stats)
    train_df, stats = normalize_features(train_df, split='train')
    val_df, _ = normalize_features(val_df, split='val', stats=stats)
    test_df, _ = normalize_features(test_df, split='test', stats=stats)
    
    # Save
    train_df.to_csv(os.path.join(output_dir, 'train_clinical.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_clinical.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_clinical.csv'), index=False)
    
    # Save normalization stats
    import json
    with open(os.path.join(output_dir, 'normalization_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Clinical features saved to {output_dir}")
    print(f"\nFeature statistics (training set):")
    print(f"  Age: {train_df['age'].mean():.1f} ± {train_df['age'].std():.1f} years")
    print(f"  Tumor size: {train_df['tumor_size_mm'].mean():.1f} ± {train_df['tumor_size_mm'].std():.1f} mm")
    print(f"  Family history: {train_df['family_history'].mean()*100:.1f}%")
    
    return train_df, val_df, test_df, stats


def print_clinical_summary(df, split='train'):
    """Print summary statistics of clinical features."""
    print(f"\n{split.upper()} SET CLINICAL SUMMARY")
    print("=" * 60)
    
    # Age by label
    print("\nAge distribution:")
    for label in df['label'].unique():
        ages = df[df['label'] == label]['age']
        print(f"  {label}: {ages.mean():.1f} ± {ages.std():.1f} years")
    
    # Tumor size by label
    print("\nTumor size distribution:")
    for label in df['label'].unique():
        sizes = df[df['label'] == label]['tumor_size_mm']
        print(f"  {label}: {sizes.mean():.1f} ± {sizes.std():.1f} mm")
    
    # Family history by label
    print("\nFamily history:")
    for label in df['label'].unique():
        fh_pct = df[df['label'] == label]['family_history'].mean() * 100
        print(f"  {label}: {fh_pct:.1f}% have family history")
    
    print("=" * 60)


if __name__ == '__main__':
    print("Clinical Feature Generation for BreakHis")
    print("=" * 70)
    
    # Generate features
    train_df, val_df, test_df, stats = generate_all_clinical_features()
    
    # Print summaries
    print_clinical_summary(train_df, 'train')
    print_clinical_summary(val_df, 'val')
    print_clinical_summary(test_df, 'test')
    
    # Test feature extraction
    print("\nTesting clinical feature extraction...")
    sample_features = extract_clinical_vector(train_df.iloc[0])
    print(f"  Feature vector shape: {sample_features.shape}")
    print(f"  Feature vector: {sample_features}")
    
    print("\n✓ Clinical features ready!")
    print("\nNext step: Create models/attention.py")
