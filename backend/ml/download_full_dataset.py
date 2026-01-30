import os
import subprocess
import shutil
from pathlib import Path

def download_and_setup_data():
    # Setup paths
    base_dir = Path('data/raw')
    images_dir = base_dir / 'images'
    
    # Ensure directories exist
    base_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    os.environ['KAGGLE_CONFIG_DIR'] = os.path.abspath('config')
    
    print("Starting full dataset download (this may take a while)...")
    
    try:
        # Download and unzip
        subprocess.run([
            "kaggle", "datasets", "download", 
            "nih-chest-xrays/data", 
            "-p", str(base_dir),
            "--unzip"
        ], check=True)
        print("Download and extraction complete.")
        
    except subprocess.CalledProcessError as e:
        print(f"Error during download: {e}")
        return

    print("Organizing files...")
    
    # The dataset typically unzips into subfolders like images_001, images_002, etc.
    # OR it might unzip directly. We need to handle both.
    # Based on file list, likely: images_001/images/file.png
    
    # Find all png files in subdirectories and move them to data/raw/images
    count = 0
    for root, dirs, files in os.walk(base_dir):
        # Skip the target directory itself to avoid recursion issues if we were running this multiple times
        if Path(root) == images_dir:
            continue
            
        for file in files:
            if file.lower().endswith('.png'):
                src_path = Path(root) / file
                dst_path = images_dir / file
                
                # Move file
                shutil.move(str(src_path), str(dst_path))
                count += 1
                
                if count % 1000 == 0:
                    print(f"Moved {count} images...", end='\r')
    
    print(f"\nSuccessfully moved {count} images to {images_dir}")
    
    # Cleanup empty folders
    # (Optional, but good for cleanliness. Be careful not to delete raw data if move failed)
    # create a list of directories to remove (images_*)
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.startswith('images_') and item.name != 'images':
            print(f"Cleaning up {item.name}...")
            shutil.rmtree(item)

    print("Data setup complete.")

if __name__ == "__main__":
    download_and_setup_data()
