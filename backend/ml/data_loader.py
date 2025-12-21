import os
import pandas as pd
from pathlib import Path

def get_data_paths():
    """
    Returns paths for raw data, processed data, and the metadata CSV.
    """
    # Resolve paths relative to this file (backend/ml/data_loader.py) -> root/data
    # backend/ml/data_loader.py -> parent = backend/ml -> parent = backend -> parent = root
    root_dir = Path(__file__).resolve().parent.parent.parent
    base_path = root_dir / 'data'
    
    raw_dir = base_path / 'raw'
    processed_dir = base_path / 'processed'
    images_dir = raw_dir / 'images'
    csv_path = raw_dir / 'Data_Entry_2017.csv'
    
    return {
        'raw_dir': raw_dir,
        'processed_dir': processed_dir,
        'images_dir': images_dir,
        'csv_path': csv_path
    }

def load_pneumonia_dataset(base_dir):
    """
    Scans the directory for NORMAL and PNEUMONIA folders.
    Expected structure:
    base_dir/
      train/
        NORMAL/
        PNEUMONIA/
      test/
      val/
    
    Returns a DataFrame with [image_path, label, split]
    label: 0 for NORMAL, 1 for PNEUMONIA
    """
    data = []
    
    # We look for chest_xray folder if it exists, otherwise assume we are in it
    search_dir = base_dir
    possible_sub = base_dir / 'chest_xray'
    if possible_sub.exists():
        search_dir = possible_sub
        
    print(f"Scanning for data in: {search_dir}")
    
    for split in ['train', 'test', 'val']:
        split_dir = search_dir / split
        if not split_dir.exists():
            print(f"Warning: Split directory not found: {split_dir}")
            continue
            
        for class_name, label in [('NORMAL', 0), ('PNEUMONIA', 1)]:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                print(f"Warning: Class directory not found: {split}/{class_name}")
                continue
                
            # Scan for images
            for img_path in class_dir.glob('*.jpeg'): # Dataset uses .jpeg
                data.append({
                    'image_path': str(img_path),
                    'label': label,
                    'split': split
                })
            # Also check for .png just in case
            for img_path in class_dir.glob('*.png'):
                data.append({
                    'image_path': str(img_path),
                    'label': label,
                    'split': split
                })
                
    if not data:
        # Fallback: maybe they flattened it? Scan recursively
        print("Standard structure not found, scanning recursively...")
        for img_path in search_dir.rglob('*'):
            if img_path.suffix.lower() in ['.jpeg', '.jpg', '.png']:
                # Guess label from parent folder name
                parent = img_path.parent.name.upper()
                grandparent = img_path.parent.parent.name.lower()
                
                label = 1 if 'PNEUMONIA' in parent else 0
                
                # Guess split from folder path
                split = 'train'
                if 'test' in str(img_path).lower():
                    split = 'test'
                elif 'val' in str(img_path).lower():
                    split = 'val'
                    
                data.append({
                    'image_path': str(img_path),
                    'label': label,
                    'split': split
                })

    return pd.DataFrame(data)
