import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import combinations
from collections import Counter
import os

def perform_eda():
    # Setup paths
    base_dir = Path('data/raw')
    csv_path = base_dir / 'Data_Entry_2017.csv'
    reports_dir = Path('reports/figures')
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return

    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    print(df.head())

    # 1. Processing Labels
    print("Processing labels...")
    # 'Finding Labels' contains pipe-separated labels (e.g., "Cardiomegaly|Emphysema")
    
    # Get all unique labels
    all_labels = []
    for labels in df['Finding Labels']:
        for label in labels.split('|'):
            if label != 'No Finding':
                all_labels.append(label)
    
    label_counts = Counter(all_labels)
    label_df = pd.DataFrame.from_dict(label_counts, orient='index', columns=['Count']).sort_values('Count', ascending=False)
    
    print("\nDisease Counts:")
    print(label_df)

    # 2. Plot Class Distribution
    plt.figure(figsize=(12, 8))
    sns.barplot(x=label_df['Count'], y=label_df.index, palette='viridis')
    plt.title('Distribution of Findings in NIH Chest X-ray Dataset')
    plt.xlabel('Number of Images')
    plt.ylabel('Finding')
    plt.tight_layout()
    plt.savefig(reports_dir / 'class_distribution.png')
    print(f"Saved class distribution plot to {reports_dir / 'class_distribution.png'}")
    plt.close()

    # 3. Patient Level Analysis
    print("\nAnalyzing Patient Distribution...")
    if 'Patient ID' in df.columns:
        images_per_patient = df.groupby('Patient ID').size()
        print(f"Total unique patients: {len(images_per_patient)}")
        print(f"Average images per patient: {images_per_patient.mean():.2f}")
        print(f"Max images per patient: {images_per_patient.max()}")
        
        plt.figure(figsize=(10, 6))
        plt.hist(images_per_patient, bins=50, color='skyblue', edgecolor='black', log=True)
        plt.title('Distribution of Images per Patient (Log Scale)')
        plt.xlabel('Number of Images')
        plt.ylabel('Number of Patients (Log)')
        plt.tight_layout()
        plt.savefig(reports_dir / 'images_per_patient.png')
        print(f"Saved patient distribution plot to {reports_dir / 'images_per_patient.png'}")
        plt.close()
    else:
        print("Warning: 'Patient ID' column not found.")

    # 4. Co-occurrence Matrix
    print("\nGenerating Co-occurrence Matrix...")
    # Create binary columns for each disease
    unique_diseases = sorted(label_df.index.tolist())
    
    # Initialize matrix
    co_matrix = pd.DataFrame(0, index=unique_diseases, columns=unique_diseases)
    
    for labels in df['Finding Labels']:
        current_labels = [l for l in labels.split('|') if l != 'No Finding']
        if len(current_labels) > 1:
            for l1, l2 in combinations(current_labels, 2):
                co_matrix.loc[l1, l2] += 1
                co_matrix.loc[l2, l1] += 1  # Verify symmetry
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(co_matrix, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Co-occurrence Matrix of Diseases')
    plt.tight_layout()
    plt.savefig(reports_dir / 'co_occurrence_matrix.png')
    print(f"Saved co-occurrence matrix to {reports_dir / 'co_occurrence_matrix.png'}")
    plt.close()

    print("\nEDA Complete.")

if __name__ == "__main__":
    perform_eda()
