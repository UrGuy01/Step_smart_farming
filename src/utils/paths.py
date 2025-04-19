import os
import sys
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Data paths
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Results paths
VISUALIZATIONS_DIR = os.path.join(RESULTS_DIR, 'visualizations')
REPORTS_DIR = os.path.join(RESULTS_DIR, 'reports')

# Files
CROP_RECOMMENDATION_FILE = os.path.join(RAW_DATA_DIR, 'Crop_recommendation.csv')
EXTENDED_CROP_RECOMMENDATION_FILE = os.path.join(RAW_DATA_DIR, 'Crop Recommendation using Soil Properties and Weather Prediction.csv')

# Create directories if they don't exist
def ensure_dir_exists(directory):
    """Create directory if it doesn't exist"""
    if isinstance(directory, str):
        directory = Path(directory)
    
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    return directory

# Ensure all directories exist
def ensure_all_dirs():
    dirs = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        VISUALIZATIONS_DIR,
        REPORTS_DIR
    ]
    for d in dirs:
        ensure_dir_exists(d)

# Data files
CROP_RECOMMENDATION_CSV = os.path.join(RAW_DATA_DIR, "Crop_recommendation.csv")

# Processed data files
X_TRAIN_NPY = os.path.join(PROCESSED_DATA_DIR, "X_train.npy")
X_TEST_NPY = os.path.join(PROCESSED_DATA_DIR, "X_test.npy")
Y_TRAIN_NPY = os.path.join(PROCESSED_DATA_DIR, "y_train.npy")
Y_TEST_NPY = os.path.join(PROCESSED_DATA_DIR, "y_test.npy")
FEATURE_NAMES_NPY = os.path.join(PROCESSED_DATA_DIR, "feature_names.npy")
CLASS_NAMES_NPY = os.path.join(PROCESSED_DATA_DIR, "class_names.npy")

# Model files
MODEL_PKL = os.path.join(MODELS_DIR, "crop_recommendation_model.pkl")
SCALER_PKL = os.path.join(PROCESSED_DATA_DIR, "scaler.pkl")
LABEL_ENCODER_PKL = os.path.join(PROCESSED_DATA_DIR, "label_encoder.pkl")

# Raw data files
CROP_RECOMMENDATION_EXTENDED_CSV = os.path.join(RAW_DATA_DIR, "Crop Recommendation using Soil Properties and Weather Prediction.csv")

# Processed data files
ORIGINAL_DATASET_CSV = os.path.join(PROCESSED_DATA_DIR, "original_dataset.csv")
NEW_DATASET_CSV = os.path.join(PROCESSED_DATA_DIR, "new_dataset.csv")
EXTENDED_CROP_DATA_CSV = os.path.join(PROCESSED_DATA_DIR, "extended_crop_data.csv")
PREPROCESSED_COMBINED_CROP_DATA_CSV = os.path.join(PROCESSED_DATA_DIR, "preprocessed_combined_crop_data.csv")
PROCESSED_CSV = os.path.join(PROCESSED_DATA_DIR, "processed_crop_data.csv")

# Training data files
X_TRAIN_COMBINED_NPY = os.path.join(PROCESSED_DATA_DIR, "X_train_combined.npy")
X_TEST_COMBINED_NPY = os.path.join(PROCESSED_DATA_DIR, "X_test_combined.npy")
Y_TRAIN_COMBINED_NPY = os.path.join(PROCESSED_DATA_DIR, "y_train_combined.npy")
Y_TEST_COMBINED_NPY = os.path.join(PROCESSED_DATA_DIR, "y_test_combined.npy")

# Extended dataset training files
X_TRAIN_EXTENDED_NPY = os.path.join(PROCESSED_DATA_DIR, "X_train_extended.npy")
X_TEST_EXTENDED_NPY = os.path.join(PROCESSED_DATA_DIR, "X_test_extended.npy")
Y_TRAIN_EXTENDED_NPY = os.path.join(PROCESSED_DATA_DIR, "y_train_extended.npy")
Y_TEST_EXTENDED_NPY = os.path.join(PROCESSED_DATA_DIR, "y_test_extended.npy")

# Model files
BASIC_MODEL_PKL = os.path.join(MODELS_DIR, "random_forest_model.pkl")
COMBINED_MODEL_PKL = os.path.join(MODELS_DIR, "random_forest_combined.pkl")
EXTENDED_MODEL_PKL = os.path.join(MODELS_DIR, "random_forest_extended.pkl")

# Preprocessor and encoder files
SCALER_COMBINED_PKL = os.path.join(PROCESSED_DATA_DIR, "scaler_combined.pkl")
LABEL_ENCODER_COMBINED_PKL = os.path.join(PROCESSED_DATA_DIR, "label_encoder_combined.pkl")
EXTENDED_PREPROCESSOR_PKL = os.path.join(PROCESSED_DATA_DIR, "extended_preprocessor.pkl")
SOIL_COLOR_ENCODER_PKL = os.path.join(PROCESSED_DATA_DIR, "soil_color_encoder.pkl")

# Metadata files
FEATURE_STATS_CSV = os.path.join(PROCESSED_DATA_DIR, "feature_stats.csv")
FEATURE_STATS_COMBINED_CSV = os.path.join(PROCESSED_DATA_DIR, "feature_stats_combined.csv")
EXTENDED_FEATURE_NAMES_CSV = os.path.join(PROCESSED_DATA_DIR, "extended_feature_names.csv")
LABEL_MAPPING_CSV = os.path.join(PROCESSED_DATA_DIR, "label_mapping.csv")

def get_project_root():
    """Return the project root directory"""
    # Assuming this script is in src/utils/
    return Path(__file__).parent.parent.parent

def get_data_dir():
    """Return the data directory"""
    return get_project_root() / 'data'

def get_raw_data_dir():
    """Return the raw data directory"""
    return get_data_dir() / 'raw'

def get_processed_data_dir():
    """Return the processed data directory"""
    return get_data_dir() / 'processed'

def get_models_dir():
    """Return the models directory"""
    return get_project_root() / 'models'

def get_results_dir():
    """Return the results directory"""
    return get_project_root() / 'results'

def get_visualizations_dir():
    """Return the visualizations directory"""
    return get_results_dir() / 'visualizations'

def get_reports_dir():
    """Return the reports directory"""
    return get_results_dir() / 'reports'

def join_paths(*paths):
    """Join paths in a cross-platform way"""
    return os.path.join(*paths)

if __name__ == "__main__":
    ensure_all_dirs()
    print(f"Project root: {project_root}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Raw data directory: {RAW_DATA_DIR}")
    print(f"Processed data directory: {PROCESSED_DATA_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Visualizations directory: {VISUALIZATIONS_DIR}")
    print(f"Reports directory: {REPORTS_DIR}") 