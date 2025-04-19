import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.field_image_utils import overlay_label_on_image

# Set page config
st.set_page_config(
    page_title="Smart Farming Field Analyzer",
    page_icon="🌱",
    layout="wide"
)

class BasicCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(BasicCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

@st.cache_resource
def load_model(model_path, model_type='basic', num_classes=9):
    """Load the trained model"""
    device = torch.device('cpu')
    
    if model_type == 'resnet':
        model = models.resnet18()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_type == 'mobilenet':
        model = models.mobilenet_v2()
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    else:  # basic model
        model = BasicCNN(num_classes=num_classes)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess the image for model prediction"""
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB (in case it's RGBA)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transformations
    img_tensor = transform(image)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor

def predict_image(model, image, class_names):
    """Make prediction on the image"""
    # Preprocess the image
    img_tensor = preprocess_image(image)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    # Get the predicted class index
    predicted_class_idx = torch.argmax(probabilities[0]).item()
    
    # Get the class name
    predicted_class = class_names[predicted_class_idx]
    
    # Get the confidence score
    confidence = float(probabilities[0][predicted_class_idx])
    
    return predicted_class, confidence, probabilities[0].numpy()

def display_prediction(image, class_names, predictions, predicted_class):
    """Display the prediction results"""
    # Create two columns
    col1, col2 = st.columns(2)
    
    # Display the image in the first column
    with col1:
        st.image(image, caption='Uploaded Field Image', use_column_width=True)
    
    # Display the prediction results in the second column
    with col2:
        st.subheader("Prediction Results")
        st.write(f"**Predicted Condition:** {predicted_class}")
        
        # Show confidence score for the predicted class
        confidence = float(predictions[np.argmax(predictions)])
        st.write(f"**Confidence:** {confidence:.2%}")
        
        # Display a progress bar for the confidence
        st.progress(confidence)
        
        # Display a bar chart for all class probabilities
        st.subheader("Class Probabilities")
        fig, ax = plt.subplots(figsize=(10, 5))
        y_pos = np.arange(len(class_names))
        ax.barh(y_pos, predictions, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(class_names)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Probability')
        ax.set_title('Class Probabilities')
        
        # Remove the chart background
        ax.grid(alpha=0.3)
        fig.tight_layout()
        
        st.pyplot(fig)

def display_field_condition_info(predicted_class):
    """Display information about the predicted field condition"""
    # Define information about each field condition
    field_condition_info = {
        "double_plant": {
            "description": "Double planting occurs when seeds are planted too close together, leading to competition for resources.",
            "impact": "Reduced yield potential due to competition for water, nutrients, and sunlight.",
            "action": "Adjust planter settings for proper seed spacing in future plantings. Monitor affected areas for potential reduced yield."
        },
        "drydown": {
            "description": "Drydown is the process of crop moisture reduction before harvest.",
            "impact": "Natural process, but irregular drydown patterns may indicate uneven field conditions.",
            "action": "Monitor moisture levels for optimal harvest timing. Consider field variability in future management decisions."
        },
        "endrow": {
            "description": "End rows are the areas at the ends of the field where equipment turns around.",
            "impact": "Often shows compaction, irregular seed placement, or reduced plant health.",
            "action": "Consider reduced tillage or targeted soil amendments to address compaction issues in these areas."
        },
        "nutrient_deficiency": {
            "description": "Plants showing signs of lacking essential nutrients required for healthy growth.",
            "impact": "Reduced crop vigor, yield potential, and possibly quality.",
            "action": "Conduct soil tests to identify specific deficiencies. Apply targeted fertilizer or amendments based on test results."
        },
        "planter_skip": {
            "description": "Areas where the planter failed to place seeds, resulting in gaps in the crop rows.",
            "impact": "Direct yield loss in areas with missing plants. May allow increased weed pressure.",
            "action": "Check planter mechanisms and settings before future plantings. Consider weed control in affected areas."
        },
        "storm_damage": {
            "description": "Damage to crops from severe weather events such as wind, hail, or heavy rain.",
            "impact": "Can range from minor leaf damage to complete crop destruction, depending on severity.",
            "action": "Assess the extent of damage. Contact crop insurance if significant. Consider management options like replanting if early season."
        },
        "water": {
            "description": "Standing water or flooding in the field.",
            "impact": "Root damage, oxygen deprivation, nutrient leaching, and potential disease problems.",
            "action": "Evaluate drainage solutions for the affected areas. Monitor for disease development as fields dry."
        },
        "waterway": {
            "description": "Natural or constructed channels designed to handle water flow through a field.",
            "impact": "Necessary for water management, but may reduce planted area.",
            "action": "Maintain grassy waterways to prevent erosion. Ensure they're functioning properly for water movement."
        },
        "weed_cluster": {
            "description": "Concentrated areas of weed growth within the crop.",
            "impact": "Competition for resources, potential yield loss, and seed bank development for future problems.",
            "action": "Implement targeted weed control measures. Consider adjusting herbicide program for the specific weeds identified."
        }
    }
    
    # Check if the predicted class exists in our info dictionary
    if predicted_class in field_condition_info:
        info = field_condition_info[predicted_class]
        
        st.subheader("Field Condition Information")
        
        st.markdown(f"**Description:** {info['description']}")
        st.markdown(f"**Potential Impact:** {info['impact']}")
        st.markdown(f"**Recommended Action:** {info['action']}")
    else:
        st.write("No detailed information available for this field condition.")

def main():
    # Main app header
    st.title("🌱 Smart Farming Field Analyzer")
    st.markdown("Upload field images to identify conditions and get management recommendations")
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Model selection
    model_dir = 'models/field_classifier'
    
    # Debug: Print the absolute path
    print(f"Looking for models in: {os.path.abspath(model_dir)}")

    if not os.path.exists(model_dir):
        st.error(f"Model directory not found. Please train a model first.")
        model_path = None
        model_type = None
    else:
        # Get list of model subdirectories
        model_dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
        
        # Debug: Print available model directories
        print(f"Available model directories: {model_dirs}")

        if not model_dirs:
            st.error("No trained model available. Please train a model before using this application.")
            model_path = None
            model_type = None
        else:
            selected_model = st.sidebar.selectbox(
                "Select a trained model",
                model_dirs,
                index=0
            )
            
            # Extract model type from directory name (assuming format: type_timestamp)
            model_type = selected_model.split('_')[0] if '_' in selected_model else 'basic'
            
            model_path = os.path.join(model_dir, selected_model, 'best_model.pth')
            
            # Check if model file exists
            if not os.path.exists(model_path):
                st.sidebar.error(f"Model file not found at {model_path}")
                model_path = None
    
    # Class names (these should match the classes the model was trained on)
    class_names = [
        'double_plant', 'drydown', 'endrow', 'nutrient_deficiency',
        'planter_skip', 'storm_damage', 'water', 'waterway', 'weed_cluster'
    ]
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a field image...", type=["jpg", "jpeg", "png"])
    
    # Check if file is uploaded and model is available
    if uploaded_file and model_path:
        # Load the model
        model = load_model(model_path, model_type=model_type, num_classes=len(class_names))
        
        # Display spinner while processing
        with st.spinner("Processing image..."):
            # Load and display the image
            image = Image.open(uploaded_file)
            
            # Make prediction
            predicted_class, confidence, predictions = predict_image(model, image, class_names)
            
            # Display prediction
            display_prediction(image, class_names, predictions, predicted_class)
            
            # Add a separator
            st.markdown("---")
            
            # Display information about the field condition
            display_field_condition_info(predicted_class)
    
    # If no file is uploaded, show sample images
    elif not uploaded_file:
        st.info("Please upload a field image to get started.")
        
        # Add example images section
        st.markdown("### Example Images")
        st.markdown("Below are some examples of different field conditions:")
        
        # Display example images in a grid (if available)
        example_dir = os.path.join('data', 'examples')
        if os.path.exists(example_dir):
            example_images = [f for f in os.listdir(example_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            if example_images:
                cols = st.columns(3)
                for i, img_name in enumerate(example_images[:6]):  # Show up to 6 examples
                    with cols[i % 3]:
                        img_path = os.path.join(example_dir, img_name)
                        img = Image.open(img_path)
                        condition = img_name.split('_')[0]  # Assume filename format: condition_*.jpg
                        st.image(img, caption=f"{condition.replace('_', ' ').title()}", use_column_width=True)
        
    # Show a warning if model is not available
    elif not model_path:
        st.error("No trained model available. Please train a model before using this application.")
        
        # Add instructions on how to train a model
        st.markdown("""
        ### How to Train a Model
        
        To use this application, you need to first train a field classifier model using the following steps:
        
        1. Process the field image dataset: 
           ```
           python src/preprocessing/process_field_dataset.py --data_dir your_data_dir --output_dir data/processed/field_dataset
           ```
        
        2. Train a field classifier model:
           ```
           python src/training/field_classifier.py --dataset_dir data/processed/field_dataset --model_type resnet
           ```
        
        Once training is complete, the model will be available in the dropdown menu.
        """)

if __name__ == "__main__":
    main() 