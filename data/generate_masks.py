
import os
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import cv2


def generate_mask_from_image(image_path, label):
    img = np.array(Image.open(image_path).convert('RGB'))
    H, W = img.shape[:2]
    
    if label == 'benign':
        # Benign: no tumor regions
        return np.zeros((H, W), dtype=np.uint8)
    
    elif label == 'malignant':
        blue = img[:, :, 2]
        red = img[:, :, 0]
        purple_score = blue.astype(float) - red.astype(float) * 0.5
        purple_score = np.clip(purple_score, 0, 255).astype(np.uint8)
        _, binary = cv2.threshold(purple_score, 100, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = (binary > 0).astype(np.uint8)
        return mask
    
    else:
        raise ValueError(f"Unknown label: {label}")


def generate_all_masks(data_dir='./data/processed', output_dir='./data/masks'):
    import pandas as pd
    
    os.makedirs(output_dir, exist_ok=True)
    
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
            
            if not os.path.exists(image_path):
                print(f"WARNING: {image_path} not found, skipping")
                continue
            
            mask = generate_mask_from_image(image_path, label)
            
            mask_filename = filename.replace('.png', '_mask.npy')
            mask_path = os.path.join(split_output, mask_filename)
            np.save(mask_path, mask)
        
        print(f"✓ Generated {len(df)} masks for {split}")
    
    print(f"\n✓ All masks saved to {output_dir}")


def visualize_samples(data_dir='./data/processed', mask_dir='./data/masks', n_samples=5):
    import pandas as pd
    import matplotlib.pyplot as plt
    
    train_csv = os.path.join(data_dir, 'train.csv')
    if not os.path.exists(train_csv):
        print(f"ERROR: {train_csv} not found")
        return
    
    df = pd.read_csv(train_csv)
    
    malignant_df = df[df['label'] == 'malignant'].sample(n=min(n_samples, len(df)))
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
    
    for i, (idx, row) in enumerate(malignant_df.iterrows()):
        image_path = row['image_path']
        filename = row['filename']
        
        img = Image.open(image_path).convert('RGB')
        
        mask_filename = filename.replace('.png', '_mask.npy')
        mask_path = os.path.join(mask_dir, 'train', mask_filename)
        
        if not os.path.exists(mask_path):
            print(f"WARNING: {mask_path} not found")
            continue
        
        mask = np.load(mask_path)
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original\n{filename}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title("Generated Mask")
        axes[i, 1].axis('off')
        
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
    generate_all_masks()
    print("\nGenerating visualization...")
    try:
        visualize_samples()
    except Exception as e:
        print(f"Visualization failed (expected if no display): {e}")
    
    print("\n✓ Mask generation complete!")
    print("\nNext step: Create models/unet_segmenter.py")
