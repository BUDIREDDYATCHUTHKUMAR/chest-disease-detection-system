import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms

# Import local modules
try:
    from model import ChestXrayModel
except ImportError:
    from backend.ml.model import ChestXrayModel

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        
        # Register hooks
        # We handle backward hook inside forward hook to ensure we track the right tensor
        target_layer.register_forward_hook(self.save_activation)
        
    def save_activation(self, module, input, output):
        self.activation = output
        # Register backward hook directly on the tensor
        output.register_hook(self.save_gradient)
        
    def save_gradient(self, grad):
        self.gradients = grad
        
    def __call__(self, x, class_idx=None):
        # 1. Forward Pass
        self.model.eval()
        
        # We need zero_grad to clear previous gradients
        self.model.zero_grad()
        
        output = self.model(x)
        
        if class_idx is None:
            # Default to the predicted class with highest score
            class_idx = output.argmax(dim=1).item()
            
        # 2. Backward Pass
        self.model.zero_grad()
        target = output[0][class_idx]
        target.backward()
        
        # 3. Generate Map
        # Global Average Pooling of Gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the channels by gradients
        activation = self.activation[0] # (C, H, W)
        for i in range(activation.shape[0]):
            activation[i, :, :] *= pooled_gradients[i]
            
        # Average the channels (Heatmap)
        heatmap = torch.mean(activation, dim=0).cpu().detach().numpy()
        
        # ReLU (only positive contributions)
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
            
        return heatmap, output

def overlay_heatmap(heatmap, original_image_path, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlays heatmap on the original image.
    original_image_path: Path to the image file or PIL Image
    """
    if isinstance(original_image_path, (str, Path)):
        img = cv2.imread(str(original_image_path))
    else:
        img = np.array(original_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Colorize
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Overlay
    overlay = cv2.addWeighted(heatmap_colored, alpha, img, 1 - alpha, 0)
    
    # Convert back to RGB for Grid/Matplotlib
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Grad-CAM Heatmap')
    parser.add_argument('--image_path', type=str, help='Path to input image')
    parser.add_argument('--output_path', type=str, default='gradcam_output.png', help='Path to save output')
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained model weights')
    args = parser.parse_args()
    
    print("Initializing Grad-CAM...")
    
    # Load Model
    model = ChestXrayModel(num_classes=14, pretrained=True)
    if args.model_path and Path(args.model_path).exists():
        print(f"Loading weights from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()
    
    target_layer = model.model.features.norm5
    grad_cam = GradCAM(model, target_layer)
    
    if args.image_path:
        img_path = Path(args.image_path)
        if not img_path.exists():
            print(f"Error: Image {img_path} not found.")
            exit(1)
            
        print(f"Processing {img_path}...")
        
        # Preprocess
        # Use same transforms as training (resize + normalize)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        raw_image = Image.open(img_path).convert('RGB')
        input_tensor = transform(raw_image).unsqueeze(0)
        
        # Run Grad-CAM
        heatmap, logits = grad_cam(input_tensor)
        
        # Get Top Prediction
        probs = torch.sigmoid(logits)[0]
        top_idx = probs.argmax().item()
        
        # Overlay
        # Resize raw image to 224x224 for overlay consistency or keep original?
        # Let's resize raw image to match heatmap size logic (224x224)
        raw_resized = raw_image.resize((224, 224))
        result = overlay_heatmap(heatmap, np.array(raw_resized))
        
        # Save
        cv2.imwrite(args.output_path, result)
        print(f"Saved Grad-CAM visualization to {args.output_path}")
        
    else:
        # Dummy Test
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        heatmap, _ = grad_cam(x)
        print("Dummy run successful.")
