import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np

class ChestXrayDataset(Dataset):
    """
    Custom Dataset for NIH Chest X-ray Database
    """
    def __init__(self, csv_file, image_dir, labels, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            image_dir (string): Directory with all the images.
            labels (list): List of column names to be used as target labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.labels = labels
        self.transform = transform
        
        # Verify required columns exist
        if 'Image Index' not in self.data.columns:
            raise ValueError("CSV must contain 'Image Index' column")
            
        for label in self.labels:
            if label not in self.data.columns:
                raise ValueError(f"Label '{label}' not found in CSV columns")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir, self.data.iloc[idx]['Image Index'])
        
        try:
            image = Image.open(img_name).convert('RGB')
        except (IOError, FileNotFoundError) as e:
            # Handle missing images gracefully or raise error
            print(f"Error loading image {img_name}: {e}")
            # Return a blank image or handle appropriately. 
            # For training, crashing is usually better to know something is wrong.
            raise e

        # Get labels
        # We need to extract the values for the specified label columns
        labels = self.data.iloc[idx][self.labels].values.astype(np.float32)
        labels = torch.from_numpy(labels)

        if self.transform:
            image = self.transform(image)

        return image, labels
