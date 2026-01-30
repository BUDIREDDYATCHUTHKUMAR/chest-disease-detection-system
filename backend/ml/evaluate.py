import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
from pathlib import Path
from tqdm import tqdm

# Import local modules
try:
    from dataset import ChestXrayDataset
    from model import ChestXrayModel
    from train import get_label_columns
except ImportError:
    from backend.ml.dataset import ChestXrayDataset
    from backend.ml.model import ChestXrayModel
    from backend.ml.train import get_label_columns

def evaluate_model(model_path=None, batch_size=32, max_batches=None):
    # Paths
    root_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = root_dir / 'data'
    processed_dir = data_dir / 'processed'
    images_dir = data_dir / 'raw' / 'images'
    models_dir = root_dir / 'models'
    
    if model_path is None:
        model_path = models_dir / 'best_model.pth'
        if not model_path.exists():
            model_path = models_dir / 'final_model.pth'
            
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        return

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Test Data
    test_df_path = processed_dir / 'test.csv'
    if not test_df_path.exists():
        print("Error: test.csv not found.")
        return
        
    print("Loading test dataframe...")
    test_df = pd.read_csv(test_df_path)
    
    # Identify Labels (Same logic as train)
    labels = get_label_columns(test_df)
    print(f"Evaluating on {len(labels)} labels: {labels}")
    
    # Transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Dataset & DataLoader
    test_dataset = ChestXrayDataset(test_df_path, images_dir, labels, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Load Model
    print(f"Loading model from {model_path}...")
    model = ChestXrayModel(num_classes=len(labels), pretrained=False) # No need to download weights again
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Inference
    all_targets = []
    all_preds = []
    
    print("Running inference...")
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            for batch_idx, (images, targets) in enumerate(tepoch):
                if max_batches and batch_idx >= max_batches:
                    break
                    
                images = images.to(device)
                outputs = model(images)
                
                # Apply sigmoid since we used BCEWithLogitsLoss during training
                preds = torch.sigmoid(outputs)
                
                all_targets.append(targets.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                
    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)
    
    # Calculate Metrics
    print("\nCalculating metrics...")
    
    # AUC-ROC per class
    try:
        aucs = []
        for i, label in enumerate(labels):
            try:
                score = roc_auc_score(all_targets[:, i], all_preds[:, i])
                aucs.append(score)
                print(f"{label}: {score:.4f}")
            except ValueError:
                print(f"{label}: Could not calculate AUC (only one class present in batch?)")
                
        if aucs:
            print(f"\nMean AUC: {np.mean(aucs):.4f}")
    except Exception as e:
        print(f"Error calculating AUC: {e}")
        
    return all_targets, all_preds

if __name__ == "__main__":
    evaluate_model()
