import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.data_utils import (
    load_crop_recommendation_data, 
    load_extended_crop_recommendation_data,
    preprocess_combined_data,
    preprocess_extended_features
)
from src.utils.model_utils import train_random_forest_model, evaluate_model
from src.utils.paths import MODELS_DIR, RESULTS_DIR, ensure_all_dirs

def main():
    """Main function to train the combined and extended crop prediction models"""
    # Ensure all directories exist
    ensure_all_dirs()
    
    # Load datasets
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
    
    # Fix column names in the extended dataset to match the basic dataset
    print("\nProcessing extended dataset column names...")
    
    # Rename columns to match basic dataset
    column_mapping = {
        'Ph': 'ph',
        'N': 'N',
        'P': 'P',
        'K': 'K',
        'label': 'label'
    }
    
    # Add missing columns with default values based on averages from each season
    extended_df['temperature'] = (extended_df['T2M_MAX-W'] + extended_df['T2M_MAX-Sp'] + 
                                 extended_df['T2M_MAX-Su'] + extended_df['T2M_MAX-Au'] + 
                                 extended_df['T2M_MIN-W'] + extended_df['T2M_MIN-Sp'] + 
                                 extended_df['T2M_MIN-Su'] + extended_df['T2M_MIN-Au']) / 8
    
    extended_df['humidity'] = (extended_df['QV2M-W'] + extended_df['QV2M-Sp'] + 
                              extended_df['QV2M-Su'] + extended_df['QV2M-Au']) * 10
    
    extended_df['rainfall'] = (extended_df['PRECTOTCORR-W'] + extended_df['PRECTOTCORR-Sp'] + 
                              extended_df['PRECTOTCORR-Su'] + extended_df['PRECTOTCORR-Au']) * 25
    
    # Rename the columns
    extended_df = extended_df.rename(columns=column_mapping)
    
    # Process combined dataset (using common features)
    print("\nProcessing combined dataset...")
    X_train_combined, X_test_combined, y_train_combined, y_test_combined, scaler_combined, label_encoder_combined = preprocess_combined_data(basic_df, extended_df)
    
    print("\nCombined preprocessing complete:")
    print(f"Combined training data shape: {X_train_combined.shape}")
    print(f"Combined testing data shape: {X_test_combined.shape}")
    
    # Train combined model
    print("\nTraining Random Forest model on combined dataset...")
    combined_model = train_random_forest_model(
        X_train_combined, y_train_combined, 
        n_estimators=100, 
        random_state=42,
        model_name='random_forest_combined'
    )
    
    # Evaluate combined model
    print("\nEvaluating combined model...")
    combined_accuracy, combined_report = evaluate_model(
        combined_model, X_test_combined, y_test_combined, 
        label_encoder_combined, 
        output_prefix='combined_'
    )
    
    print("\nCombined Model performance metrics:")
    print(f"Accuracy: {combined_accuracy:.4f}")
    print("\nClassification Report:")
    print(combined_report)
    
    # Process extended dataset with all features
    print("\nProcessing extended dataset with all features...")
    X_train_ext, X_test_ext, y_train_ext, y_test_ext, preprocessor_ext, label_encoder_ext = preprocess_extended_features(extended_df)
    
    print("\nExtended preprocessing complete:")
    print(f"Extended training data shape: {X_train_ext.shape}")
    print(f"Extended testing data shape: {X_test_ext.shape}")
    
    # Train extended model
    print("\nTraining Random Forest model on extended dataset...")
    extended_model = train_random_forest_model(
        X_train_ext, y_train_ext, 
        n_estimators=100, 
        random_state=42,
        model_name='random_forest_extended'
    )
    
    # Evaluate extended model
    print("\nEvaluating extended model...")
    extended_accuracy, extended_report = evaluate_model(
        extended_model, X_test_ext, y_test_ext, 
        label_encoder_ext, 
        output_prefix='extended_'
    )
    
    print("\nExtended Model performance metrics:")
    print(f"Accuracy: {extended_accuracy:.4f}")
    print("\nClassification Report:")
    print(extended_report)
    
    print(f"\nModel training completed:")
    print(f"Combined model saved to {MODELS_DIR}/random_forest_combined.pkl")
    print(f"Extended model saved to {MODELS_DIR}/random_forest_extended.pkl")
    print(f"Evaluation metrics saved to {RESULTS_DIR}")
    
    # Compare model performance
    print("\nModel Performance Comparison:")
    print(f"Combined Model Accuracy: {combined_accuracy:.4f}")
    print(f"Extended Model Accuracy: {extended_accuracy:.4f}")
    
    if extended_accuracy > combined_accuracy:
        print("The Extended Model performs better!")
    elif combined_accuracy > extended_accuracy:
        print("The Combined Model performs better!")
    else:
        print("Both models perform equally well.")

if __name__ == "__main__":
    main() 