import os
import pandas as pd
from pathlib import Path

def get_data_paths():
    """
    Returns paths for raw data, processed data, and the metadata CSV.
    """
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

def load_metadata(csv_path):
    """
    Loads the Data_Entry_2017.csv file.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata CSV not found at {csv_path}")
    
    return pd.read_csv(csv_path)

def map_images(df, images_dir):
    """
    Verifies that images listed in the CSV exist in the images directory.
    """
    # Look for both .png and .jpg
    df['image_path'] = df['Image Index'].apply(lambda x: os.path.join(images_dir, x))
    df['exists'] = df['image_path'].apply(os.path.exists)
    
    full_count = len(df)
    missing_count = full_count - df['exists'].sum()
    
    if missing_count > 0:
        print(f"WARNING: {missing_count} images listed in CSV are missing from {images_dir}")
        
    return df[df['exists']].copy()
