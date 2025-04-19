import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.data_utils import load_crop_recommendation_data, preprocess_basic_data
from src.utils.model_utils import train_random_forest_model, evaluate_model
from src.utils.paths import MODELS_DIR, RESULTS_DIR, ensure_all_dirs

def main():
    """Main function to train the basic crop prediction model"""
    # Ensure all directories exist
    ensure_all_dirs()
    
    print("Loading crop recommendation dataset...")
    df = load_crop_recommendation_data()
    
    if df is None:
        print("Failed to load dataset. Exiting.")
        return
    
    print(f"Dataset loaded with shape: {df.shape}")
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler, label_encoder = preprocess_basic_data(df)
    
    print("\nPreprocessing complete:")
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Train model
    print("\nTraining Random Forest model...")
    model = train_random_forest_model(
        X_train, y_train, 
        n_estimators=100, 
        random_state=42,
        model_name='random_forest_model'
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    accuracy, report = evaluate_model(model, X_test, y_test, label_encoder)
    
    print("\nModel performance metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    print(f"\nModel training completed. Model saved to {MODELS_DIR}/random_forest_model.pkl")
    print(f"Evaluation metrics saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main() 