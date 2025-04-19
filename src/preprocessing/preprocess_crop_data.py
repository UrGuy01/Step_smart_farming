import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.data_utils import load_crop_recommendation_data, preprocess_basic_data
from src.utils.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, VISUALIZATIONS_DIR, ensure_all_dirs

def main():
    """Main function to preprocess the basic crop recommendation dataset"""
    # Ensure all directories exist
    ensure_all_dirs()
    
    print("Loading crop recommendation dataset...")
    df = load_crop_recommendation_data()
    
    if df is None:
        print("Failed to load dataset. Exiting.")
        return
    
    print(f"Dataset loaded with shape: {df.shape}")
    
    # Display dataset info
    print("\nDataset Information:")
    print(f"Number of samples: {df.shape[0]}")
    print(f"Number of features: {df.shape[1] - 1}")  # Exclude the target column
    print(f"Features: {', '.join(df.columns[:-1])}")
    print(f"Target: {df.columns[-1]}")
    
    # Display class distribution
    print("\nClass Distribution:")
    class_counts = df['label'].value_counts()
    for crop, count in class_counts.items():
        print(f"{crop}: {count} samples ({count/len(df)*100:.2f}%)")
    
    # Generate class distribution visualization
    plt.figure(figsize=(12, 6))
    sns.countplot(x='label', data=df)
    plt.title('Crop Distribution')
    plt.xlabel('Crop Type')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.tight_layout()
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'crop_distribution.png'))
    plt.close()
    
    # Generate pairplot for features
    print("\nGenerating feature visualizations (this may take a moment)...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.drop('label', axis=1).corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'feature_correlation.png'))
    plt.close()
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler, label_encoder = preprocess_basic_data(df)
    
    print("\nPreprocessing complete:")
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    print(f"\nAll preprocessing steps completed. Files saved to {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    main() 