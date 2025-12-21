import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_disease_distributions(df, labels):
    """
    Plots the frequency of each pathology in the dataset.
    """
    counts = df[labels].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=counts.values, y=counts.index, palette='viridis')
    plt.title('Distribution of Thoracic Diseases')
    plt.xlabel('Number of Cases')
    plt.ylabel('Pathology')
    plt.tight_layout()
    plt.show()

def plot_cooccurrence(df, labels):
    """
    Plots the co-occurrence matrix of diseases.
    """
    matrix = df[labels].T.dot(df[labels])
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Reds')
    plt.title('Disease Co-occurrence Matrix')
    plt.tight_layout()
    plt.show()
