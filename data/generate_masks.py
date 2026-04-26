"""
Generate Synthetic Segmentation Masks for BreakHis

Since BreakHis doesn't have pixel-level tumor annotations, we generate
synthetic masks based on image characteristics:
- Malignant: High-density purple regions (nuclei)
- Benign: No tumor regions (all background)

This is a simplification, but works for multi-task learning.
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import cv2


def generate_mask_from_image(image_path, label):
    """
    Generate synthetic segmentation mask from histopathology image.
    
    Strategy:
    - Benign: return all zeros (no tumor)
    - Malignant: threshold purple/blue channels (nuclei), apply morphology
    
    Args:
        image_path: path to RGB image
        label: 'benign' or 'malignant'
    
    Returns:
        mask: (H, W) numpy array, 0=background, 1=tumor
    """
    # Load image
    img = np.array(Image.open(image_path).convert('RGB'))
    H, W = img.shape[:2]
    
    if label == 'benign':
        # Benign: no tumor regions
        return np.zeros((H, W), dtype=np.uint8)
    
    elif label == 'malignant':
        # Malignant: detect dense nuclear regions
        
        # H&E staining: nuclei are purple/blue
        # Extract blue channel (high for nuclei)
        blue = img[:, :, 2]
        
        # Also use red-blue difference (purple = high blue, low red)
        red = img[:, :, 0]
        purple_score = blue.astype(float) - red.astype(float) * 0.5
        purple_score = np.clip(purple_score, 0, 255).astype(np.uint8)
        
        # Threshold to get nuclear regions
        _, binary = cv2.threshold(purple_score, 100, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Close small holes
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Remove small isolated regions
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Convert to 0/1
        mask = (binary > 0).astype(np.uint8)
        
        return mask
    
    else:
        raise ValueError(f"Unknown label: {label}")


def generate_all_masks(data_dir='./data/processed', output_dir='./data/masks'):
    """
    Generate masks for all images in the dataset.
    
    Args:
        data_dir: directory with train.csv, val.csv, test.csv
        output_dir: where to save generated masks
    """
    import pandas as pd
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each split
    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(data_dir, f'{split}.csv')
        
        if not os.path.exists(csv_path):
            print(f"WARNING: {csv_path} not found, skipping {split}")
            continue
        
        df = pd.read_csv(csv_path)
        print(f"\nGenerating masks for {split} set ({len(df)} images)...")
        
        split_output = os.path.join(output_dir, split)
        os.makedirs(split_output, exist_ok=True)
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=split):
            image_path = row['image_path']
            label = row['label']
            filename = row['filename']
            
            # Check if image exists
            if not os.path.exists(image_path):
                print(f"WARNING: {image_path} not found, skipping")
                continue
            
            # Generate mask
            mask = generate_mask_from_image(image_path, label)
            
            # Save mask
            mask_filename = filename.replace('.png', '_mask.npy')
            mask_path = os.path.join(split_output, mask_filename)
            np.save(mask_path, mask)
        
        print(f"✓ Generated {len(df)} masks for {split}")
    
    print(f"\n✓ All masks saved to {output_dir}")


def visualize_samples(data_dir='./data/processed', mask_dir='./data/masks', n_samples=5):
    """
    Visualize sample images with generated masks.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Load train data
    train_csv = os.path.join(data_dir, 'train.csv')
    if not os.path.exists(train_csv):
        print(f"ERROR: {train_csv} not found")
        return
    
    df = pd.read_csv(train_csv)
    
    # Sample malignant images
    malignant_df = df[df['label'] == 'malignant'].sample(n=min(n_samples, len(df)))
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
    
    for i, (idx, row) in enumerate(malignant_df.iterrows()):
        image_path = row['image_path']
        filename = row['filename']
        
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Load mask
        mask_filename = filename.replace('.png', '_mask.npy')
        mask_path = os.path.join(mask_dir, 'train', mask_filename)
        
        if not os.path.exists(mask_path):
            print(f"WARNING: {mask_path} not found")
            continue
        
        mask = np.load(mask_path)
        
        # Plot
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original\n{filename}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title("Generated Mask")
        axes[i, 1].axis('off')
        
        # Overlay
        img_np = np.array(img)
        overlay = img_np.copy()
        overlay[mask == 1] = overlay[mask == 1] * 0.5 + np.array([255, 0, 0]) * 0.5
        overlay = overlay.astype(np.uint8)
        
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Overlay (red = tumor)")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('mask_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to mask_visualization.png")
    plt.show()


if __name__ == '__main__':
    print("Synthetic Mask Generation for BreakHis")
    print("=" * 70)
    
    # Generate masks
    generate_all_masks()
    
    # Visualize
    print("\nGenerating visualization...")
    try:
        visualize_samples()
    except Exception as e:
        print(f"Visualization failed (expected if no display): {e}")
    
    print("\n✓ Mask generation complete!")
    print("\nNext step: Create models/unet_segmenter.py")
