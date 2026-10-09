import os
import numpy as np
import pandas as pd
from pathlib import Path


def simulate_clinical_features(df, seed=42):
    np.random.seed(seed)
    
    n_samples = len(df)
    df = df.copy()
    
    ages = []
    for label in df['label']:
        if label == 'benign':
            age = np.random.normal(45, 12)
        else:  # malignant
            age = np.random.normal(58, 15)
        
        age = np.clip(age, 20, 90)
        ages.append(age)
    
    df['age'] = ages
    
    sizes = []
    for label in df['label']:
        if label == 'benign':
            # Benign: smaller (mean 15mm, std 8)
            size = np.random.normal(15, 8)
        else:
            # Malignant: larger (mean 28mm, std 12)
            size = np.random.normal(28, 12)
        
        size = np.clip(size, 5, 50)
        sizes.append(size)
    
    df['tumor_size_mm'] = sizes
    
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
    
    magnifications = df['magnification'].unique()
    for mag in magnifications:
        df[f'mag_{mag}'] = (df['magnification'] == mag).astype(int)
    
    return df


def normalize_features(df, split='train', stats=None):
    continuous_features = ['age', 'tumor_size_mm']
    
    if split == 'train':
        stats = {}
        for feat in continuous_features:
            stats[feat] = {
                'mean': df[feat].mean(),
                'std': df[feat].std()
            }
        
        for feat in continuous_features:
            df[f'{feat}_norm'] = (df[feat] - stats[feat]['mean']) / stats[feat]['std']
    
    else:
        if stats is None:
            raise ValueError("Must provide training statistics for val/test normalization")
        
        for feat in continuous_features:
            df[f'{feat}_norm'] = (df[feat] - stats[feat]['mean']) / stats[feat]['std']
    
    return df, stats


def extract_clinical_vector(row):
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
    os.makedirs(output_dir, exist_ok=True)
    
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    print("Generating clinical features...")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    train_df = simulate_clinical_features(train_df, seed=42)
    val_df = simulate_clinical_features(val_df, seed=43)
    test_df = simulate_clinical_features(test_df, seed=44)
    
    train_df, stats = normalize_features(train_df, split='train')
    val_df, _ = normalize_features(val_df, split='val', stats=stats)
    test_df, _ = normalize_features(test_df, split='test', stats=stats)
    
    train_df.to_csv(os.path.join(output_dir, 'train_clinical.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_clinical.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_clinical.csv'), index=False)
    
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
    print(f"\n{split.upper()} SET CLINICAL SUMMARY")
    print("=" * 60)
    
    print("\nAge distribution:")
    for label in df['label'].unique():
        ages = df[df['label'] == label]['age']
        print(f"  {label}: {ages.mean():.1f} ± {ages.std():.1f} years")
    
    print("\nTumor size distribution:")
    for label in df['label'].unique():
        sizes = df[df['label'] == label]['tumor_size_mm']
        print(f"  {label}: {sizes.mean():.1f} ± {sizes.std():.1f} mm")
    
    print("\nFamily history:")
    for label in df['label'].unique():
        fh_pct = df[df['label'] == label]['family_history'].mean() * 100
        print(f"  {label}: {fh_pct:.1f}% have family history")
    
    print("=" * 60)


if __name__ == '__main__':
    print("Clinical Feature Generation for BreakHis")
    print("=" * 70)
    
    train_df, val_df, test_df, stats = generate_all_clinical_features()
    
    print_clinical_summary(train_df, 'train')
    print_clinical_summary(val_df, 'val')
    print_clinical_summary(test_df, 'test')
    
    print("\nTesting clinical feature extraction...")
    sample_features = extract_clinical_vector(train_df.iloc[0])
    print(f"  Feature vector shape: {sample_features.shape}")
    print(f"  Feature vector: {sample_features}")
    
    print("\n✓ Clinical features ready!")
    print("\nNext step: Create models/attention.py")
