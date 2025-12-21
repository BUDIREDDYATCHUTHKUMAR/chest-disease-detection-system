import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
try:
    from .data_loader import get_data_paths, load_pneumonia_dataset
except ImportError:
    from data_loader import get_data_paths, load_pneumonia_dataset

def clean_and_split():
    paths = get_data_paths()
    print("Initializing Pneumonia Dataset Adapter...")
    
    # Load Data using directory scanning
    df = load_pneumonia_dataset(paths['images_dir'])
    
    if len(df) == 0:
        print("ERROR: No images found! Please check data/raw/images")
        return
        
    print(f"Found {len(df)} total images.")
    print("Class distribution:\n", df['label'].value_counts())
    print("Split distribution:\n", df['split'].value_counts())
    
    # The Pneumonia dataset 'val' split is usually tiny (16 images), 
    # and 'test' is reasonable (624), 'train' is large (5216).
    # We can keep them as is, or redistribute. 
    # For now, let's respect the user's folders but handle the case if 'val' is too small.
    
    # Save splits
    save_dir = paths['processed_dir']
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    
    # Save standard CSVs
    for split_name in ['train', 'val', 'test']:
        split_df = df[df['split'] == split_name].copy()
        
        # We rename columns to match what our future model might expect (generic 'label')
        # And keep image_path
        output_path = save_dir / f'{split_name}.csv'
        split_df.to_csv(output_path, index=False)
        print(f"Saved {split_name} set to {output_path} ({len(split_df)} images)")

    print(f"Data processing complete. Ready for training.")
    return df

if __name__ == "__main__":
    clean_and_split()
