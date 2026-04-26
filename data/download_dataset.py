"""
BreakHis Dataset Download & Organization Script

This script:
1. Downloads the BreakHis dataset from Kaggle
2. Organizes it into a clean folder structure
3. Creates metadata CSV for easy loading

Run this FIRST before anything else.

Requirements:
    pip install kaggle opendatasets --break-system-packages

Usage:
    python data/download_dataset.py
"""

import os
from pathlib import Path

DATASET_PATH = Path(r"C:\Users\bbbsa\Downloads\multimodal-cancer-COMPLETE\multimodal-cancer-detection\data")


def download_breakhis():
    """
    Download BreakHis dataset from Kaggle.

    NOTE: You need a Kaggle account and API key.

    Steps to get Kaggle API key:
    1. Go to https://www.kaggle.com/settings
    2. Click "Create New API Token"
    3. Download kaggle.json
    4. Place it in ~/.kaggle/ (Linux/Mac) or C:\\Users\\YourName\\.kaggle\\ (Windows)
    """
    print("Downloading BreakHis dataset from Kaggle...")
    print("This will take 5-10 minutes (~700 MB).\n")

    try:
        import opendatasets as od

        dataset_url = 'https://www.kaggle.com/datasets/ambarish/breakhis'
        od.download(dataset_url, data_dir='./data/raw')

        print("\nDownload complete!")
        return True

    except ImportError:
        print("ERROR: opendatasets not installed.")
        print("Install with: pip install opendatasets --break-system-packages")
        return False
    except Exception as e:
        print(f"ERROR downloading dataset: {e}")
        print("\nManual download instructions:")
        print("1. Go to: https://www.kaggle.com/datasets/ambarish/breakhis")
        print("2. Click 'Download' button")
        print("3. Extract to: ./data/raw/breakhis/")
        return False


def organize_dataset():
    """
    BreakHis comes in a nested folder structure. This flattens it for easier access.

    We'll create a simpler CSV with:
        image_path, label, magnification, subtype
    """
    print("\nOrganizing dataset structure...")

    base_path = Path("./data/raw/breakhis/BreaKHis_v1/histology_slides/breast")

    if not base_path.exists():
        print(f"ERROR: Dataset not found at {base_path}")
        print("Make sure download completed successfully.")
        return False

    image_data = []

    for main_class in ['benign', 'malignant']:
        sob_path = base_path / main_class / 'SOB'

        if not sob_path.exists():
            print(f"WARNING: {sob_path} not found, skipping...")
            continue

        for subtype_dir in sob_path.iterdir():
            if not subtype_dir.is_dir():
                continue

            subtype = subtype_dir.name

            for img_path in subtype_dir.rglob('*.png'):
                magnification = img_path.parent.name
                patient_id = img_path.parent.parent.name

                try:
                    relative_path = img_path.relative_to(Path.cwd())
                except ValueError:
                    relative_path = img_path.resolve()

                image_data.append({
                    'image_path': str(relative_path),
                    'label': main_class,
                    'magnification': magnification,
                    'subtype': subtype,
                    'patient_id': patient_id,
                    'filename': img_path.name,
                })

    if not image_data:
        print("ERROR: No images were found under the extracted BreakHis folders.")
        print(f"Checked path: {base_path.resolve()}")
        return False

    print(f"Found {len(image_data)} images")
    print(f"  Benign: {sum(1 for x in image_data if x['label'] == 'benign')}")
    print(f"  Malignant: {sum(1 for x in image_data if x['label'] == 'malignant')}")

    import pandas as pd

    df = pd.DataFrame(image_data)

    os.makedirs('./data/processed', exist_ok=True)
    csv_path = './data/processed/dataset_full.csv'
    df.to_csv(csv_path, index=False)

    print(f"\nSaved dataset metadata to {csv_path}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Total images: {len(df)}")

    return True


def verify_dataset():
    """Quick sanity check that everything downloaded correctly."""
    import pandas as pd

    csv_path = './data/processed/dataset_full.csv'
    if not os.path.exists(csv_path):
        print("ERROR: dataset_full.csv not found. Run organize_dataset() first.")
        return False

    if os.path.getsize(csv_path) == 0:
        print("ERROR: dataset_full.csv is empty. Run organize_dataset() again.")
        return False

    df = pd.read_csv(csv_path)

    if df.empty:
        print("ERROR: dataset_full.csv has no rows. Run organize_dataset() again.")
        return False

    print("\n" + "=" * 50)
    print("DATASET VERIFICATION")
    print("=" * 50)
    print(f"Total images: {len(df)}")
    print(f"\nClass distribution:")
    print(df['label'].value_counts())
    print(f"\nMagnification distribution:")
    print(df['magnification'].value_counts().sort_index())
    print(f"\nSubtype distribution:")
    print(df['subtype'].value_counts())

    sample_paths = df['image_path'].sample(min(10, len(df))).tolist()
    missing = [p for p in sample_paths if not os.path.exists(p)]

    if missing:
        print(f"\nWARNING: {len(missing)} sample images not found:")
        for p in missing[:3]:
            print(f"  - {p}")
    else:
        print("\nAll sample images verified")

    print("=" * 50)
    return True


if __name__ == '__main__':
    print("BreakHis Dataset Setup")
    print("=" * 50)

    if not os.path.exists('./data/raw/breakhis'):
        success = download_breakhis()
        if not success:
            print("\nDownload failed. Please download manually and try again.")
            exit(1)
    else:
        print("Dataset already downloaded. Skipping download step.")

    if not os.path.exists('./data/processed/dataset_full.csv'):
        success = organize_dataset()
        if not success:
            print("\nOrganization failed. Check error messages above.")
            exit(1)
    else:
        print("Dataset already organized. Skipping organization step.")

    verify_dataset()

    print("\nDataset setup complete!")
    print("\nNext step: Run data/preprocess.py to create train/val/test splits")
