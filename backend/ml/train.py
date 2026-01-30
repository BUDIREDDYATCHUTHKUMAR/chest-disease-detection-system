import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
from sklearn.metrics import roc_auc_score

# Import local modules
try:
    from dataset import ChestXrayDataset
    from model import ChestXrayModel
except ImportError:
    from backend.ml.dataset import ChestXrayDataset
    from backend.ml.model import ChestXrayModel

def get_label_columns(df):
    """
    Identify label columns dynamically.
    We assume labels are the columns that are not in the metadata list.
    """
    metadata_cols = [
        'Image Index', 'Finding Labels', 'Follow-up #', 'Patient ID', 
        'Patient Age', 'Patient Gender', 'View Position', 
        'OriginalImage[Width', 'Height]', 'OriginalImagePixelSpacing[x', 'y]', 
        'Unnamed: 0', 'image_path', 'exists'
    ]
    
    potential_labels = []
    for col in df.columns:
        if col not in metadata_cols and pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() <= 2 and not df[col].isnull().any(): 
                potential_labels.append(col)
            
    return sorted(potential_labels)

def train_model(num_epochs=10, batch_size=16, learning_rate=1e-4, max_batches=None):
    # Paths
    root_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = root_dir / 'data'
    processed_dir = data_dir / 'processed'
    images_dir = data_dir / 'raw' / 'images'
    models_dir = root_dir / 'models'
    
    if not models_dir.exists():
        models_dir.mkdir(parents=True)

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Dataframes
    train_df_path = processed_dir / 'train.csv'
    val_df_path = processed_dir / 'val.csv'
    
    if not train_df_path.exists():
        print("Error: train.csv not found. Run clean_data.py first.")
        return

    print("Loading dataframes...")
    train_df = pd.read_csv(train_df_path)
    val_df = pd.read_csv(val_df_path)
    
    labels = get_label_columns(train_df)
    print(f"Identified {len(labels)} labels: {labels}")
    
    # Advanced Data Augmentation for Medical Imaging
    # ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), # Increased rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Increased jitter
        transforms.ToTensor(),
        normalize
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Datasets
    train_dataset = ChestXrayDataset(train_df_path, images_dir, labels, transform=train_transform)
    val_dataset = ChestXrayDataset(val_df_path, images_dir, labels, transform=val_transform)
    
    # DataLoaders (reduce num_workers for Windows stability if needed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model
    print("Initializing DenseNet121 model...")
    model = ChestXrayModel(num_classes=len(labels), pretrained=True)
    model = model.to(device)
    
    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2) # Monitor AUC (max)
    
    # Training Loop
    print(f"Starting training for {num_epochs} epochs...")
    best_val_auc = 0.0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        
        # Training Phase
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, (images, targets) in enumerate(tepoch):
                if max_batches and batch_idx >= max_batches:
                    break
                images = images.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                tepoch.set_postfix(loss=loss.item())
                
        epoch_loss = running_loss / len(train_dataset)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_targets = []
        val_preds = []
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                if max_batches and batch_idx >= max_batches:
                    break
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                
                # Store for AUC calculation
                val_targets.append(targets.cpu().numpy())
                val_preds.append(torch.sigmoid(outputs).cpu().numpy())
                
        val_loss = val_loss / len(val_dataset)
        
        # Calculate AUC
        val_targets = np.vstack(val_targets)
        val_preds = np.vstack(val_preds)
        
        try:
            val_auc = roc_auc_score(val_targets, val_preds, average="macro")
        except:
            val_auc = 0.5 # Fallback if calculation fails
            
        print(f"Epoch {epoch+1} done in {time.time() - start_time:.0f}s | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        # Scheduler Step
        scheduler.step(val_auc)
        
        # Save Best Model Logic
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), models_dir / 'best_model.pth')
            print(f"--> Saved new best model (AUC: {val_auc:.4f})")
            
    # Save final model
    torch.save(model.state_dict(), models_dir / 'final_model.pth')
    print("Training complete.")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Chest X-ray Model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()
    
    train_model(num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
