# Project Trainer's Guide

This document is designed to give Project Trainers and Developers a deep understanding of the **Thoracic Disease Detection System**.

## 1. System Architecture

The application is built as a monolithic repository with three distinct layers:

### A. Backend (Django + DRF)
- **Location**: `backend/`
- **Purpose**: Handles API requests, manages patient records, and serves as the bridge to the ML model.
- **Key Files**:
    - `manage.py`: Django entry point.
    - `backend/urls.py`: Main API routing.
    - `ml/`: Custom app for Machine Learning logic.

### B. Machine Learning (PyTorch/Scikit-Learn)
- **Location**: `backend/ml/`
- **Purpose**: Data processing, model training, and inference.
- **Workflow**:
    1.  **Ingestion**: `data_loader.py` reads raw images and CSV from `data/root`.
    2.  **Processing**: `clean_data.py` splits data into Train/Val/Test and handles multi-label binarization.
    3.  **Training**: (Future) `train.py` will load processed data and train the CNN.
    4.  **Inference**: Django views will import the trained model to predict diseases on new images.

### C. Frontend (React + Tailwind)
- **Location**: `frontend/`
- **Purpose**: User interface for doctors to upload X-rays and view predictions.
- **Communication**: Calls Django APIs at `http://localhost:8000/api/`.

## 2. Data Pipeline Explained

The project strictly follows the **NIH Chest X-ray Dataset** format.

1.  **Raw Data**:
    - `data/raw/Data_Entry_2017.csv`: Contains metadata (Patient ID, Labels).
    - `data/raw/images/`: Flat directory of all 100k+ images.

2.  **Cleaning Logic (`clean_data.py`)**:
    - Parses "Finding Labels" (e.g., "Cardiomegaly|Effusion") into binary columns (0/1).
    - Splits data: 80% Train, 10% Val, 10% Test.
    - Saves splits to `data/processed/`.

## 3. Key Commands for Trainers

### Setup Verification
Always start by ensuring the environment is healthy.
```bash
# Check Python
python --version
# Check Packages
pip show torch django pandas
```

### Running the Full Stack
Use Docker for a consistent "production-like" run:
```bash
docker-compose up --build
```
- Frontend: [http://localhost:3000](http://localhost:3000)
- Backend: [http://localhost:8000](http://localhost:8000)

### Debugging ML Scripts
To test the ML logic in isolation (without running the full server):
```bash
cd backend/ml
python clean_data.py
```

## 4. Common Pitfalls
- **Missing Data**: If `clean_data.py` fails, check `data/raw/images`. Users often forget to flatten the subfolders.
- **Path Issues**: The project relies on `data/` being at the root. Do not move the `data` folder inside `backend`.
