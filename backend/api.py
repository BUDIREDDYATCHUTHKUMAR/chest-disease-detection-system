from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from PIL import Image
import io
from pathlib import Path
import sys
import random 

# Auth imports
from backend import models, auth, database
from pydantic import BaseModel

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

try:
    import torch
    from torchvision import transforms
    from backend.ml.model import ChestXrayModel
except ImportError:
    print("WARNING: ML dependencies not found. Running in MOCK mode.")
    torch = None
    transforms = None

    class ChestXrayModel:
        def __init__(self, num_classes, pretrained=False):
            self.num_classes = num_classes
        def to(self, device): return self
        def eval(self): pass
        def load_state_dict(self, state_dict): pass
        def __call__(self, x):
            probs = [random.random() for _ in range(self.num_classes)]
            total = sum(probs)
            probs = [p / total for p in probs]
            class MockOutput:
                def __init__(self, probs): self.probs = probs
                def __getitem__(self, key): return self.probs[key]
                def tolist(self): return self.probs
            return MockOutput(probs)
    
    if torch is None:
        print("Using Mock ChestXrayModel.")

# Define clean labels list
LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 
    'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

app = FastAPI(title="Chest Disease Prediction API")

# Setup Database Tables
models.Base.metadata.create_all(bind=database.engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
if torch:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"

def get_model():
    try:
        from backend.ml.model import ChestXrayModel
    except ImportError:
        from ml.model import ChestXrayModel
    return ChestXrayModel

@app.on_event("startup")
async def load_model():
    global model
    if torch is None:
        try:
             model = ChestXrayModel(num_classes=len(LABELS))
        except Exception as e:
             print(f"Failed to init mock model: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    
    try:
        ModelClass = get_model()
        print("Model class imported.")
        model = ModelClass(num_classes=len(LABELS), pretrained=False)
        print("Model instance created.")
        
        root_dir = Path(__file__).resolve().parent.parent
        best_model_path = root_dir / 'models' / 'best_model.pth'
        final_model_path = root_dir / 'models' / 'final_model.pth'
        
        load_path = next((p for p in [best_model_path, final_model_path] if p.exists()), None)
            
        if load_path:
            print(f"Loading weights from {load_path}...")
            state_dict = torch.load(load_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Model loaded from {load_path}")
        else:
            print("WARNING: No trained model found. Predictions will be random.")
            
        print(f"Moving model to {device}...")
        model.to(device)
        model.eval()
        print("Model verification complete.")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()

# AUTH ENDPOINTS
@app.post("/register")
def register(user: UserCreate, db: Session = Depends(database.get_db)):
    try:
        db_user = db.query(models.User).filter(models.User.username == user.username).first()
        if db_user:
            raise HTTPException(status_code=400, detail="Username already registered")
            
        db_email = db.query(models.User).filter(models.User.email == user.email).first()
        if db_email:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        hashed_password = auth.get_password_hash(user.password)
        new_user = models.User(username=user.username, email=user.email, hashed_password=hashed_password)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return {"message": "User created successfully"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Registration Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = auth.create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(current_user: models.User = Depends(auth.get_current_user)):
    return {"username": current_user.username, "email": current_user.email, "id": current_user.id}


@app.get("/")
def read_root():
    return {"message": "Chest Disease Prediction API is running"}

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None, "device": str(device)}

@app.post("/predict")
async def predict(file: UploadFile = File(...), current_user: models.User = Depends(auth.get_current_user)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not active")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        heatmap_b64 = None
        
        if torch is None: # Mock Logic
            import random, hashlib
            image_hash = hashlib.md5(contents).hexdigest()
            seed_val = int(image_hash[:8], 16)
            random.seed(seed_val)
            probs = [random.random() for _ in LABELS]
            total = sum(probs)
            probs = [p/total for p in probs]
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ])
            img_tensor = preprocess(image).unsqueeze(0).to(device)
            
            # Generate Grad-CAM if model allows
            try:
                # Re-import locally to avoid circular deps if needed or ensure it's available
                from backend.ml.gradcam import GradCAM, overlay_heatmap
                import cv2
                import numpy as np
                import base64
                
                # Check if model has the expected structure
                # DenseNet features are usually in model.model.features
                if hasattr(model, 'model') and hasattr(model.model, 'features'):
                    target_layer = model.model.features.norm5
                    grad_cam = GradCAM(model, target_layer)
                    
                    # We need to handle the forward pass carefully since GradCAM does its own
                    # But our predict logic is simple, so we can let GradCAM handle it or run it separately
                    # Let's run GradCAM to get both heatmap and logits
                    heatmap, logits = grad_cam(img_tensor)
                    
                    probs = torch.sigmoid(logits)[0].tolist()
                    
                    # Create Overlay
                    # Resize original image to 224x224 for consistent overlay
                    raw_resized = image.resize((224, 224))
                    overlay = overlay_heatmap(heatmap, np.array(raw_resized))
                    
                    # Encode to Base64
                    _, buffer = cv2.imencode('.png', overlay)
                    heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                else:
                    # Fallback for other kinds of models or if structure differs
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        probs = torch.sigmoid(outputs)[0].tolist()
            except Exception as e:
                print(f"Grad-CAM Error: {e}")
                # Fallback to standard prediction
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = torch.sigmoid(outputs)[0].tolist()
            
        results = {label: float(prob) for label, prob in zip(LABELS, probs)}
        sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
        
        return {
            "prediction": sorted_results,
            "top_finding": list(sorted_results.keys())[0],
            "top_probability": list(sorted_results.values())[0],
            "heatmap": heatmap_b64
        }
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
