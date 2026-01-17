# AI-Driven Chest Radiograph Diagnostics & Thoracic Disease Detection System

## Overview
A production-ready healthcare AI software system for detecting thoracic diseases from chest X-rays.

## Project Structure
- `backend/`: FastAPI + ML Inference
- `frontend/`: React + TailwindCSS UI
- `data/`: Dataset storage
- `notebooks/`: EDA and model experiments
- `docker/`: Container configurations

## Getting Started

### 1. Backend (FastAPI)
```bash
# Activate Virtual Env (Windows)
.\venv\Scripts\Activate.ps1

# Install Dependencies
pip install -r backend/requirements.txt

# Run Server
uvicorn backend.api:app --reload
```

### 2. Frontend (React)
```bash
cd frontend
npm install
npm run dev
```

### 3. Docker (Optional)
```bash
docker-compose up --build
```

### 4. For Trainers
See [TRAINER_GUIDE.md](TRAINER_GUIDE.md) for a deep dive into the project architecture and data pipeline.

## Phases
1. Environment Setup (Complete)
2. Dataset Ingestion (Complete)
3. Data Processing (Complete)
4. Model Development (Complete)
5. Explainable AI (Complete)
6. Backend Dev (Complete)
7. Frontend Dev (Complete)
8. Integration (Complete)
9. Deployment
10. Documentation
8. Integration
9. Deployment
10. Documentation
