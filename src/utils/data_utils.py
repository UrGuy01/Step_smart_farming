import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
from pathlib import Path

from src.utils.paths import (
    PROCESSED_DATA_DIR, SCALER_PKL, LABEL_ENCODER_PKL,
    X_TRAIN_NPY, X_TEST_NPY, Y_TRAIN_NPY, Y_TEST_NPY,
    RAW_DATA_DIR, CROP_RECOMMENDATION_FILE, EXTENDED_CROP_RECOMMENDATION_FILE,
    ensure_dir_exists
)

def check_missing_values(df):
    """Check for missing values in a DataFrame."""
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("Missing values found:")
        print(missing[missing > 0])
    else:
        print("No missing values found.")
    return missing

def plot_feature_distributions(df, features, output_file=None):
    """Plot distributions of features."""
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(features):
        plt.subplot(3, 3, i+1)
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Feature distributions saved to {output_file}")
    
    return plt.gcf()

def plot_correlation_matrix(df, output_file=None):
    """Plot correlation matrix for numerical features."""
    numeric_df = df.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation between Features')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Correlation matrix saved to {output_file}")
    
    return plt.gcf()

def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Preprocess the dataset for training:
    - Split features and target
    - Encode target labels
    - Scale features
    - Split into train/test sets
    - Save all processed data
    
    Args:
        df (pandas.DataFrame): Input dataframe with crop data
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names, class_names)
    """
    # Split features and target
    X = df.drop('label', axis=1)
    y = df['label']
    feature_names = X.columns.tolist()
    
    # Encode target labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    class_names = le.classes_
    
    # Save the label encoder
    joblib.dump(le, LABEL_ENCODER_PKL)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler
    joblib.dump(scaler, SCALER_PKL)
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Save processed data
    np.save(X_TRAIN_NPY, X_train)
    np.save(X_TEST_NPY, X_test)
    np.save(Y_TRAIN_NPY, y_train)
    np.save(Y_TEST_NPY, y_test)
    
    return X_train, X_test, y_train, y_test, feature_names, class_names

def load_processed_data():
    """
    Load preprocessed data from saved numpy files
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train = np.load(X_TRAIN_NPY)
    X_test = np.load(X_TEST_NPY)
    y_train = np.load(Y_TRAIN_NPY)
    y_test = np.load(Y_TEST_NPY)
    
    return X_train, X_test, y_train, y_test

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot and optionally save a confusion matrix
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        class_names (list): List of class names
        save_path (str): Path to save the visualization
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()

def plot_feature_importance(model, feature_names, save_path=None):
    """
    Plot and optionally save feature importance
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): List of feature names
        save_path (str): Path to save the visualization
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 8))
    plt.title('Feature Importance')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    plt.close()

def save_label_mapping(label_encoder, output_file):
    """Save the mapping between original labels and encoded values."""
    mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    
    # Create a DataFrame for the label mapping
    label_df = pd.DataFrame({
        'crop': label_encoder.classes_,
        'code': range(len(label_encoder.classes_))
    })
    
    # Save to CSV
    label_df.to_csv(output_file, index=False)
    print(f"Label mapping saved to {output_file}")
    
    return mapping

def create_preprocessed_dataset(X_scaled, y, data_source, columns, output_file):
    """Create a preprocessed dataset with features and target."""
    df_preprocessed = pd.DataFrame(X_scaled, columns=columns)
    df_preprocessed['crop'] = y
    df_preprocessed['data_source'] = data_source
    
    # Save to CSV
    df_preprocessed.to_csv(output_file, index=False)
    print(f"Preprocessed dataset saved to {output_file}")
    
    return df_preprocessed

def save_feature_stats(feature_names, scaler, output_file):
    """Save feature statistics from a fitted scaler."""
    feature_stats = pd.DataFrame({
        'feature': feature_names,
        'mean': scaler.mean_,
        'scale': scaler.scale_
    })
    
    # Save to CSV
    feature_stats.to_csv(output_file, index=False)
    print(f"Feature statistics saved to {output_file}")
    
    return feature_stats

def load_crop_recommendation_data():
    """Load the basic crop recommendation dataset"""
    try:
        return pd.read_csv(CROP_RECOMMENDATION_FILE)
    except FileNotFoundError:
        print(f"Error: File {CROP_RECOMMENDATION_FILE} not found.")
        return None

def load_extended_crop_recommendation_data():
    """Load the extended crop recommendation dataset"""
    try:
        return pd.read_csv(EXTENDED_CROP_RECOMMENDATION_FILE)
    except FileNotFoundError:
        print(f"Error: File {EXTENDED_CROP_RECOMMENDATION_FILE} not found.")
        return None

def preprocess_basic_data(df, test_size=0.2, random_state=42):
    """Preprocess the basic crop recommendation dataset"""
    # Ensure the processed data directory exists
    ensure_dir_exists(PROCESSED_DATA_DIR)
    
    # Extract features and target
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Save feature statistics for UI sliders
    feature_stats = X.describe().loc[['mean', 'std']].T.reset_index()
    feature_stats.columns = ['feature', 'mean', 'scale']
    feature_stats_path = os.path.join(PROCESSED_DATA_DIR, 'feature_stats.csv')
    feature_stats.to_csv(feature_stats_path, index=False)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encode the target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Save the scaler and label encoder
    scaler_path = os.path.join(PROCESSED_DATA_DIR, 'scaler.pkl')
    label_encoder_path = os.path.join(PROCESSED_DATA_DIR, 'label_encoder.pkl')
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, label_encoder_path)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, scaler, label_encoder

def preprocess_combined_data(basic_df, extended_df, test_size=0.2, random_state=42):
    """Process and combine both datasets based on common features"""
    # Ensure the processed data directory exists
    ensure_dir_exists(PROCESSED_DATA_DIR)
    
    # Common features in both datasets
    common_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    # Extract common features and target from basic dataset
    X_basic = basic_df[common_features]
    y_basic = basic_df['label']
    
    # Extract common features and target from extended dataset
    X_extended = extended_df[common_features]
    y_extended = extended_df['label']
    
    # Combine the datasets
    X_combined = pd.concat([X_basic, X_extended], ignore_index=True)
    y_combined = pd.concat([y_basic, y_extended], ignore_index=True)
    
    # Save feature statistics for UI sliders
    feature_stats = X_combined.describe().loc[['mean', 'std']].T.reset_index()
    feature_stats.columns = ['feature', 'mean', 'scale']
    feature_stats_path = os.path.join(PROCESSED_DATA_DIR, 'feature_stats_combined.csv')
    feature_stats.to_csv(feature_stats_path, index=False)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    
    # Encode the target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_combined)
    
    # Save the scaler and label encoder
    scaler_path = os.path.join(PROCESSED_DATA_DIR, 'scaler_combined.pkl')
    label_encoder_path = os.path.join(PROCESSED_DATA_DIR, 'label_encoder_combined.pkl')
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, label_encoder_path)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, scaler, label_encoder

def preprocess_extended_features(extended_df, test_size=0.2, random_state=42):
    """Process the extended dataset with all its features"""
    # Ensure the processed data directory exists
    ensure_dir_exists(PROCESSED_DATA_DIR)
    
    # Extract features and target
    # Assuming 'label' is the target column in the extended dataset
    features = extended_df.drop('label', axis=1)
    target = extended_df['label']
    
    # Save feature names
    feature_names = pd.DataFrame({'feature_name': features.columns})
    feature_names_path = os.path.join(PROCESSED_DATA_DIR, 'extended_feature_names.csv')
    feature_names.to_csv(feature_names_path, index=False)
    
    # Create a preprocessor for mixed data types
    # For simplicity, we'll use a combination of StandardScaler for numeric features
    # and OneHotEncoder for categorical features (if any)
    
    # Check for categorical features
    categorical_features = features.select_dtypes(include=['object', 'category']).columns
    numeric_features = features.select_dtypes(include=['int64', 'float64']).columns
    
    # Define preprocessing steps based on feature types
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ]
    )
    
    # If there are categorical features, add them to the transformer
    if len(categorical_features) > 0:
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor.transformers.append(
            ('cat', categorical_transformer, categorical_features)
        )
    
    # Fit and transform the features
    X_processed = preprocessor.fit_transform(features)
    
    # Encode the target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(target)
    
    # Save the preprocessor and label encoder
    preprocessor_path = os.path.join(PROCESSED_DATA_DIR, 'extended_preprocessor.pkl')
    label_encoder_path = os.path.join(PROCESSED_DATA_DIR, 'label_encoder_extended.pkl')
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(label_encoder, label_encoder_path)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, preprocessor, label_encoder 