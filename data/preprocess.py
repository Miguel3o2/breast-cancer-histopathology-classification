
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def extract_patient_id(filename):
    parts = filename.split('-')
    if len(parts) >= 3:
        return parts[2]  # Patient ID is the 3rd part
    return None

def create_splits(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    df['patient_id'] = df['filename'].apply(extract_patient_id)
    
    df = df.dropna(subset=['patient_id'])
    
    df['strat_key'] = df['label'] + '_' + df['magnification']
    
    patient_df = df.groupby('patient_id').agg({
        'strat_key': 'first',  # All images from same patient have same strat_key
        'label': 'first'
    }).reset_index()
    
    print(f"\nTotal unique patients: {len(patient_df)}")
    print(f"Stratification by: {patient_df['strat_key'].value_counts().to_dict()}")
    
    train_patients, temp_patients = train_test_split(
        patient_df,
        test_size=(val_ratio + test_ratio),
        stratify=patient_df['strat_key'],
        random_state=seed
    )
    
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_patients['strat_key'],
        random_state=seed
    )
    
    print(f"\nPatient splits:")
    print(f"  Train: {len(train_patients)} patients")
    print(f"  Val:   {len(val_patients)} patients")
    print(f"  Test:  {len(test_patients)} patients")
    
    train_df = df[df['patient_id'].isin(train_patients['patient_id'])].copy()
    val_df   = df[df['patient_id'].isin(val_patients['patient_id'])].copy()
    test_df  = df[df['patient_id'].isin(test_patients['patient_id'])].copy()
    
    for split_df in [train_df, val_df, test_df]:
        split_df.drop(columns=['patient_id', 'strat_key'], inplace=True)
    
    print(f"\nImage splits:")
    print(f"  Train: {len(train_df)} images ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} images ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} images ({len(test_df)/len(df)*100:.1f}%)")
    
    print(f"\nClass distribution verification:")
    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        benign_pct = (split_df['label'] == 'benign').mean() * 100
        print(f"  {name}: {benign_pct:.1f}% benign, {100-benign_pct:.1f}% malignant")
    
    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df, output_dir='./data/processed'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(f'{output_dir}/train.csv', index=False)
    val_df.to_csv(f'{output_dir}/val.csv', index=False)
    test_df.to_csv(f'{output_dir}/test.csv', index=False)
    
    print(f"\n✓ Splits saved to {output_dir}/")
    print(f"  - train.csv ({len(train_df)} images)")
    print(f"  - val.csv ({len(val_df)} images)")
    print(f"  - test.csv ({len(test_df)} images)")


def print_split_statistics(train_df, val_df, test_df):
    print("\n" + "="*60)
    print("SPLIT STATISTICS")
    print("="*60)
    
    for name, df in [('TRAIN', train_df), ('VAL', val_df), ('TEST', test_df)]:
        print(f"\n{name} SET ({len(df)} images):")
        print(f"  Label distribution:")
        for label, count in df['label'].value_counts().items():
            pct = count / len(df) * 100
            print(f"    {label}: {count} ({pct:.1f}%)")
        
        print(f"  Magnification distribution:")
        for mag, count in df['magnification'].value_counts().sort_index().items():
            pct = count / len(df) * 100
            print(f"    {mag}: {count} ({pct:.1f}%)")
    
    print("="*60)


if __name__ == '__main__':
    print("Creating Train/Val/Test Splits")
    print("="*60)
    
    csv_path = './data/processed/dataset_full.csv'
    if not Path(csv_path).exists():
        print(f"ERROR: {csv_path} not found.")
        print("Run data/download_dataset.py first.")
        exit(1)
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} images from {csv_path}")
    
    train_df, val_df, test_df = create_splits(
        df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    
    save_splits(train_df, val_df, test_df)
    
    print_split_statistics(train_df, val_df, test_df)
    
    print("\n✓ Preprocessing complete!")
    print("\nNext step: Create data/transforms.py for augmentation pipelines")
