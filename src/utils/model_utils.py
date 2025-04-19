import numpy as np
import pandas as pd
import os
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from src.utils.paths import (
    MODELS_DIR, MODEL_PKL, LABEL_ENCODER_PKL,
    SCALER_PKL, VISUALIZATIONS_DIR, RESULTS_DIR, ensure_dir_exists
)

def train_model(model, X_train, y_train):
    """
    Train a model and save it
    
    Args:
        model: Initialized model object
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training labels
    
    Returns:
        object: Trained model
    """
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Save the model
    joblib.dump(model, MODEL_PKL)
    print(f"Model saved to {MODEL_PKL}")
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model

def load_model():
    """
    Load the trained model and related preprocessing objects
    
    Returns:
        tuple: (model, scaler, label_encoder)
    """
    model = joblib.load(MODEL_PKL)
    scaler = joblib.load(SCALER_PKL)
    label_encoder = joblib.load(LABEL_ENCODER_PKL)
    
    return model, scaler, label_encoder

def evaluate_model(model, X_test, y_test, label_encoder, output_prefix=''):
    """Evaluate the model performance and save metrics to files"""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=False, zero_division=0)
    
    # Save metrics to files
    accuracy_file = os.path.join(RESULTS_DIR, f'{output_prefix}model_accuracy.txt')
    report_file = os.path.join(RESULTS_DIR, f'{output_prefix}classification_report.txt')
    
    # Save label encoder
    if label_encoder is not None:
        if output_prefix == 'combined_':
            encoder_path = os.path.join(MODELS_DIR, 'combined_label_encoder.pkl')
        else:
            encoder_path = os.path.join(MODELS_DIR, 'label_encoder.pkl')
        joblib.dump(label_encoder, encoder_path)
        print(f"Label encoder saved to {encoder_path}")
    
    # Save metrics to files
    with open(accuracy_file, 'w') as f:
        f.write(f"Model Accuracy: {accuracy:.4f}")
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    return accuracy, report

def predict_crop(features, model=None, scaler=None, label_encoder=None):
    """
    Make prediction for a single sample
    
    Args:
        features (list or array): Input features
        model (object, optional): Trained model (loaded if None)
        scaler (object, optional): Fitted scaler (loaded if None) 
        label_encoder (object, optional): Fitted label encoder (loaded if None)
    
    Returns:
        tuple: (predicted_class, probability, all_probabilities)
    """
    # Load model and preprocessing objects if not provided
    if model is None or scaler is None or label_encoder is None:
        model, scaler, label_encoder = load_model()
    
    # Convert to numpy array and reshape for single sample
    features = np.array(features).reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Get prediction and probabilities
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Get the predicted class name and its probability
    predicted_class = label_encoder.inverse_transform([prediction])[0]
    probability = probabilities[prediction]
    
    return predicted_class, probability, probabilities

def save_model_metadata(model, feature_names, class_names, metrics):
    """
    Save model metadata for reference
    
    Args:
        model: Trained model
        feature_names (list): Names of features
        class_names (list): Names of classes
        metrics (dict): Evaluation metrics
    """
    metadata_path = Path(MODELS_DIR) / "model_metadata.txt"
    
    with open(metadata_path, 'w') as f:
        f.write("Crop Recommendation Model Metadata\n")
        f.write("=" * 40 + "\n\n")
        
        # Model info
        f.write(f"Model Type: {type(model).__name__}\n")
        f.write(f"Model Parameters: {model.get_params()}\n\n")
        
        # Feature info
        f.write("Features:\n")
        for i, feature in enumerate(feature_names):
            f.write(f"  {i+1}. {feature}\n")
        f.write("\n")
        
        # Class info
        f.write("Classes:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"  {i}. {class_name}\n")
        f.write("\n")
        
        # Performance metrics
        f.write("Performance Metrics:\n")
        f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall: {metrics['recall']:.4f}\n")
        f.write(f"  F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"  Inference Time: {metrics['inference_time']:.4f} seconds\n")
    
    print(f"Model metadata saved to {metadata_path}")

def train_random_forest_model(X_train, y_train, n_estimators=100, random_state=42, model_name='random_forest_model'):
    """Train a random forest classifier"""
    # Ensure models directory exists
    ensure_dir_exists(MODELS_DIR)
    
    # Create and train the model
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Save the model
    model_path = os.path.join(MODELS_DIR, f'{model_name}.pkl')
    joblib.dump(model, model_path)
    
    return model

def load_model(model_name='random_forest_model'):
    """Load a trained model"""
    model_path = os.path.join(MODELS_DIR, f'{model_name}.pkl')
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        print(f"Model file {model_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None 