import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.data_utils import (
    load_crop_recommendation_data, 
    load_extended_crop_recommendation_data,
    preprocess_combined_data,
    preprocess_extended_features
)
from src.utils.paths import PROCESSED_DATA_DIR, VISUALIZATIONS_DIR, ensure_all_dirs

def main():
    """Main function to preprocess both the basic and extended crop recommendation datasets"""
    # Ensure all directories exist
    ensure_all_dirs()
    
    print("Loading basic crop recommendation dataset...")
    basic_df = load_crop_recommendation_data()
    
    if basic_df is None:
        print("Failed to load basic dataset. Exiting.")
        return
    
    print(f"Basic dataset loaded with shape: {basic_df.shape}")
    
    print("\nLoading extended crop recommendation dataset...")
    extended_df = load_extended_crop_recommendation_data()
    
    if extended_df is None:
        print("Failed to load extended dataset. Exiting.")
        return
    
    print(f"Extended dataset loaded with shape: {extended_df.shape}")
    
    # Display dataset info for extended dataset
    print("\nExtended Dataset Information:")
    print(f"Number of samples: {extended_df.shape[0]}")
    print(f"Number of features: {extended_df.shape[1] - 1}")  # Exclude the target column
    print(f"Features: {', '.join(extended_df.columns[:-1])}")
    print(f"Target: {extended_df.columns[-1]}")
    
    # Display class distribution for extended dataset
    print("\nExtended Dataset Class Distribution:")
    class_counts = extended_df['label'].value_counts()
    for crop, count in class_counts.items():
        print(f"{crop}: {count} samples ({count/len(extended_df)*100:.2f}%)")
    
    # Generate class distribution visualization for extended dataset
    plt.figure(figsize=(12, 6))
    sns.countplot(x='label', data=extended_df)
    plt.title('Extended Dataset Crop Distribution')
    plt.xlabel('Crop Type')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.tight_layout()
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'extended_crop_distribution.png'))
    plt.close()
    
    # Compare common features in both datasets
    common_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    print("\nComparing common features in both datasets:")
    for feature in common_features:
        basic_mean = basic_df[feature].mean()
        basic_std = basic_df[feature].std()
        ext_mean = extended_df[feature].mean()
        ext_std = extended_df[feature].std()
        
        print(f"{feature}:")
        print(f"  Basic dataset:   Mean = {basic_mean:.2f}, Std = {basic_std:.2f}")
        print(f"  Extended dataset: Mean = {ext_mean:.2f}, Std = {ext_std:.2f}")
    
    # Process combined dataset (using common features)
    print("\nProcessing combined dataset...")
    X_train_combined, X_test_combined, y_train_combined, y_test_combined, scaler_combined, label_encoder_combined = preprocess_combined_data(basic_df, extended_df)
    
    print("\nCombined preprocessing complete:")
    print(f"Combined training data shape: {X_train_combined.shape}")
    print(f"Combined testing data shape: {X_test_combined.shape}")
    
    # Process extended dataset with all features
    print("\nProcessing extended dataset with all features...")
    X_train_ext, X_test_ext, y_train_ext, y_test_ext, preprocessor_ext, label_encoder_ext = preprocess_extended_features(extended_df)
    
    print("\nExtended preprocessing complete:")
    print(f"Extended training data shape: {X_train_ext.shape}")
    print(f"Extended testing data shape: {X_test_ext.shape}")
    
    print(f"\nAll preprocessing steps completed. Files saved to {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    main() 