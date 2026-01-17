import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
try:
    from .data_loader import get_data_paths, load_metadata, map_images
except ImportError:
    from data_loader import get_data_paths, load_metadata, map_images

def clean_and_split(test_size=0.2, val_size=0.1, random_state=42):
    paths = get_data_paths()
    print("Loading metadata...")
    
    try:
        df = load_metadata(paths['csv_path'])
    except FileNotFoundError:
        print("ERROR: Data_Entry_2017.csv not found in data/raw/")
        return

    print("Mapping images...")
    df = map_images(df, paths['images_dir'])
    
    if len(df) == 0:
        print("ERROR: No valid images found matching the CSV.")
        return

    # Process Labels (Multi-label)
    print("Processing labels...")
    all_labels = set()
    for labels in df['Finding Labels']:
        for label in labels.split('|'):
            if label != 'No Finding':
                all_labels.add(label)
    
    all_labels = sorted(list(all_labels))
    print(f"Found {len(all_labels)} unique pathologies: {all_labels}")
    
    # Create Binary Columns
    for label in all_labels:
        df[label] = df['Finding Labels'].apply(lambda x: 1 if label in x else 0)
        
    # Split Data by Patient ID (Prevent Data Leakage)
    if 'Patient ID' not in df.columns:
        print("ERROR: 'Patient ID' not found in dataframe. Cannot perform patient-aware split.")
        return

    patient_ids = df['Patient ID'].unique()
    print(f"Total unique patients: {len(patient_ids)}")

    # Split patients instead of images
    train_patients, test_patients = train_test_split(patient_ids, test_size=test_size, random_state=random_state)
    train_patients, val_patients = train_test_split(train_patients, test_size=val_size / (1 - test_size), random_state=random_state)

    print(f"Patient Split: Train={len(train_patients)}, Val={len(val_patients)}, Test={len(test_patients)}")

    # Filter Dataframes based on split patient IDs
    train = df[df['Patient ID'].isin(train_patients)]
    val = df[df['Patient ID'].isin(val_patients)]
    test = df[df['Patient ID'].isin(test_patients)]
    
    # Verify no leakage
    train_p = set(train['Patient ID'])
    val_p = set(val['Patient ID'])
    test_p = set(test['Patient ID'])
    
    assert train_p.intersection(val_p) == set(), "Data Leakage detected between Train and Val!"
    assert train_p.intersection(test_p) == set(), "Data Leakage detected between Train and Test!"
    assert val_p.intersection(test_p) == set(), "Data Leakage detected between Val and Test!"

    print(f"Image Split: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    
    # Save splits
    save_dir = paths['processed_dir']
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
        
    train.to_csv(save_dir / 'train.csv', index=False)
    val.to_csv(save_dir / 'val.csv', index=False)
    test.to_csv(save_dir / 'test.csv', index=False)
    
    print(f"Data verification and cleaning complete. Saved to {save_dir}")
    return train, val, test, all_labels

if __name__ == "__main__":
    clean_and_split()
