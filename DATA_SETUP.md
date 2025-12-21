# Dataset Setup Instructions (Pneumonia Detection)

## 1. Important Note
The application is currently adapted for the **Chest X-Ray Images (Pneumonia)** dataset.

## 2. Directory Structure
Your `data` folder should look like this (nested folders are handled automatically):

```
data/
└── raw/
    └── images/
        ├── train/
        │   ├── NORMAL/
        │   └── PNEUMONIA/
        ├── test/
        │   ├── NORMAL/
        │   └── PNEUMONIA/
        └── val/
            ├── NORMAL/
            └── PNEUMONIA/
```

## 3. Processing
Run the cleaning script to generate standard CSV files for the model:
```bash
cd backend/ml
python clean_data.py
```

This will create `train.csv`, `val.csv`, and `test.csv` in `data/processed/`.
