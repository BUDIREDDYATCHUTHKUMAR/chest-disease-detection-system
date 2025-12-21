# AI-Driven Chest Radiograph Diagnostics & Thoracic Disease Detection System

## Overview
A production-ready healthcare AI software system for detecting thoracic diseases from chest X-rays.

## Project Structure
- `backend/`: Django API + ML Inference
- `frontend/`: React + TailwindCSS UI
- `data/`: Dataset storage
- `notebooks/`: EDA and model experiments
- `docker/`: Container configurations

## Getting Started

### 1. Python Virtual Environment
```bash
# Windows
.\venv\Scripts\Activate.ps1
# Mac/Linux
source venv/bin/activate

pip install -r backend/requirements.txt
```

### 2. Frontend
```bash
cd frontend
npm install
```

### 3. Docker
```bash
docker-compose up --build
```

### 4. For Trainers
See [TRAINER_GUIDE.md](TRAINER_GUIDE.md) for a deep dive into the project architecture and data pipeline.

## Phases
1. Environment Setup (Complete)
2. Dataset Ingestion (Next)
3. Data Processing
4. Model Development
5. Explainable AI
6. Backend Dev
7. Frontend Dev
8. Integration
9. Deployment
10. Documentation
