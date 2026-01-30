# AI-Driven Chest Radiograph Diagnostics & Thoracic Disease Detection System

A production-ready healthcare AI software system designed to detect thoracic diseases from chest X-rays using state-of-the-art Deep Learning models (DenseNet121).

## 🚀 Features
- **Automated Diagnosis**: Multi-label classification for 14 disease classes (e.g., Pneumonia, Effusion, Mass).
- **Explainable AI**: Grad-CAM visualization to highlight regions of interest where the model focuses.
- **Unified Pipeline**: Single-command execution for data cleaning, training, and evaluation.
- **Modern Stack**: FastAP backend, React + TailwindCSS frontend, and PyTorch for ML.

## 📂 Project Structure
- `backend/`: FastAPI application & ML core logic (`ml/`).
- `frontend/`: React application for user interaction.
- `data/`: Directory for raw and processed datasets.
- `notebooks/`: Jupyter notebooks for experiments.
- `docker/`: Docker configurations for containerization.

## 🛠️ Getting Started

### Prerequisites
- Python 3.9+
- Node.js 16+
- CUDA-enabled GPU (Optional, but recommended for training)

### 1. Backend Setup (FastAPI + ML)
1. **Create & Activate Virtual Environment**:
   ```powershell
   # Windows
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```
   > **Note**: For Python 3.13+, ensure `torch>=2.6.0` is installed.

3. **Run the API Server**:
   ```bash
   uvicorn backend.api:app --reload
   ```
   Server will run at `http://localhost:8000`.

### 2. Frontend Setup (React)
1. **Navigate to Frontend**:
   ```bash
   cd frontend
   ```

2. **Install Dependencies & Run**:
   ```bash
   npm install
   npm run dev
   ```
   App will run at `http://localhost:5173`.

## 🧠 Machine Learning Pipeline
This project features a unified pipeline script to handle the entire ML workflow.

### Run Full Pipeline
To clean data, train the model, and evaluate performance:
```bash
python backend/ml/main.py
```

### Pipeline Options
| Command | Description |
|---------|-------------|
| `--epochs <N>` | Set number of training epochs (default: 5) |
| `--batch_size <N>` | Set batch size (default: 16) |
| `--skip_clean` | Skip data cleaning/splitting step |
| `--skip_train` | Skip model training |
| `--skip_eval` | Skip model evaluation |
| `--quick_run` | Run a fast verification (1 epoch, few batches) |

**Example**:
```bash
python backend/ml/main.py --epochs 10 --batch_size 32
```

## 🐳 Docker Deployment
To run the full stack using Docker:
```bash
docker-compose up --build
```

## 📄 License
This project is open-source and available under the MIT License.
