# NIH Chest X-ray Dataset Setup

The system is configured for the **NIH Chest X-ray Dataset**.

## Required File Structure
Please ensure your `data` folder looks EXACTLY like this:

```
data/
└── raw/
    ├── Data_Entry_2017.csv  <-- The metadata CSV file
    └── images/              <-- Folder containing all .png images
        ├── 00000001_000.png
        ├── 00000001_001.png
        └── ...
```

## Instructions
1.  **Download** the dataset info and images.
2.  **Extract** `Data_Entry_2017.csv` to `data/raw/`.
3.  **Extract** all image tarballs.
4.  **Move** all images into `data/raw/images/`.
    *   *Note: Do not have subfolders like `images_001/images`. Move the files up so they are all directly inside `data/raw/images/`.*
5.  **Run Cleaning**:
    ```bash
    python -m backend.ml.clean_data
    ```
