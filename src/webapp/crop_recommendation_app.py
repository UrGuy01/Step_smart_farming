import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import preprocess_data

# Set page config
st.set_page_config(
    page_title="Smart Farming Crop Recommender",
    page_icon="🌾",
    layout="wide"
)

@st.cache_resource
def load_model(model_path):
    """Load the trained model"""
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model
    else:
        return None

def predict_crop(model, input_data, label_encoder):
    """Make prediction using the model"""
    # Make prediction
    prediction = model.predict(input_data)
    
    # Define fallback crop names mapping in case label encoder doesn't work properly
    fallback_crop_names = {
        0: "rice",
        1: "maize",
        2: "chickpea", 
        3: "kidneybeans",
        4: "pigeonpeas",
        5: "mothbeans",
        6: "mungbean",
        7: "blackgram",
        8: "lentil",
        9: "pomegranate",
        10: "banana",
        11: "mango",
        12: "grapes",
        13: "watermelon",
        14: "muskmelon",
        15: "apple",
        16: "orange",
        17: "papaya",
        18: "coconut",
        19: "cotton",
        20: "jute",
        21: "coffee"
    }
    
    # Decode the prediction - with fallback handling
    try:
        # Try using the label encoder first
        predicted_class_idx = prediction[0]
        if hasattr(label_encoder, 'classes_'):
            predicted_crop = str(label_encoder.inverse_transform(prediction)[0])
            # Check if result is numeric, and use fallback if it is
            if predicted_crop.isdigit():
                predicted_crop = fallback_crop_names.get(int(predicted_crop), f"Crop Type {predicted_crop}")
        else:
            # Use fallback if label encoder doesn't have classes attribute
            predicted_crop = fallback_crop_names.get(predicted_class_idx, f"Crop Type {predicted_class_idx}")
    except Exception as e:
        # Fallback in case of any error
        predicted_class_idx = prediction[0]
        predicted_crop = fallback_crop_names.get(predicted_class_idx, f"Crop Type {predicted_class_idx}")
    
    # Get prediction probabilities
    probabilities = model.predict_proba(input_data)[0]
    
    # Get top 3 predictions with probabilities
    top_indices = np.argsort(probabilities)[::-1][:3]
    
    # Map indices to crop names with fallback
    top_crops = []
    for idx in top_indices:
        try:
            crop_name = str(label_encoder.inverse_transform([idx])[0])
            # Check if result is numeric, and use fallback if it is
            if crop_name.isdigit():
                crop_name = fallback_crop_names.get(int(crop_name), f"Crop Type {crop_name}")
        except:
            crop_name = fallback_crop_names.get(idx, f"Crop Type {idx}")
        top_crops.append(crop_name)
    
    top_probs = probabilities[top_indices]
    
    return predicted_crop, top_crops, top_probs

def main():
    # Main app header
    st.title("🌾 Smart Farming Crop Recommender")
    st.markdown("Enter soil and climate data to get crop recommendations")
    
    # Load the model
    model_path = "models/random_forest_model.pkl"
    model = load_model(model_path)
    
    if model is None:
        st.error(f"Model not found at {model_path}. Please train the model first.")
        return
    
    # Load label encoder
    label_encoder_path = "models/label_encoder.pkl"
    if os.path.exists(label_encoder_path):
        label_encoder = joblib.load(label_encoder_path)
    else:
        # If label encoder not found, create a simple one based on the model classes
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        label_encoder.classes_ = model.classes_
    
    # Create a form for user input
    st.subheader("Enter Soil and Climate Data")
    
    with st.form("crop_recommendation_form"):
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            nitrogen = st.number_input("Nitrogen (N) content in soil (kg/ha)", 0, 150, 50)
            phosphorus = st.number_input("Phosphorus (P) content in soil (kg/ha)", 0, 150, 50)
            potassium = st.number_input("Potassium (K) content in soil (kg/ha)", 0, 150, 50)
            temperature = st.slider("Temperature (°C)", 0.0, 45.0, 25.0, 0.1)
        
        with col2:
            humidity = st.slider("Humidity (%)", 0.0, 100.0, 70.0, 0.1)
            ph = st.slider("pH value", 0.0, 14.0, 6.5, 0.1)
            rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, 100.0, 0.1)
        
        submit_button = st.form_submit_button("Get Recommendation")
    
    # Process the input when the user submits the form
    if submit_button:
        # Create input data for prediction
        input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        
        # Make prediction
        with st.spinner("Generating recommendation..."):
            predicted_crop, top_crops, top_probs = predict_crop(model, input_data, label_encoder)
        
        # Display results
        st.subheader("Recommendation Results")
        
        # Show the predicted crop
        st.markdown(f"### Recommended Crop: **{predicted_crop.title()}**")
        
        # Display top 3 recommendations with probabilities
        st.subheader("Top 3 Recommendations")
        
        # Create a bar chart for top crops
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#2ecc71', '#3498db', '#95a5a6']  # Green, Blue, Gray
        ax.bar(top_crops, top_probs * 100, color=colors)
        ax.set_ylabel('Confidence (%)')
        ax.set_title('Top Crop Recommendations')
        ax.grid(axis='y', alpha=0.3)
        
        # Add percentage labels on top of bars
        for i, (crop, prob) in enumerate(zip(top_crops, top_probs)):
            ax.text(i, prob * 100 + 1, f'{prob * 100:.1f}%', ha='center')
            
        st.pyplot(fig)
        
        # Display crop information
        st.subheader("Crop Information")
        
        # Define crop information (sample data, can be expanded)
        crop_info = {
            "rice": {
                "description": "Rice is a cereal grain and a staple food for more than half of the world's population.",
                "growing_conditions": "Prefers warm, wet conditions. Grows best in temperatures of 20-35°C with high humidity.",
                "soil_requirements": "Clay soils that can hold water well. pH between 5.5-6.5. High nitrogen requirement.",
                "water_needs": "High water requirement, often grown in flooded fields."
            },
            "maize": {
                "description": "Maize (corn) is one of the most versatile crops, used for food, feed, and industrial purposes.",
                "growing_conditions": "Warm weather crop, requires full sun exposure. Optimal temperature is 18-32°C.",
                "soil_requirements": "Well-drained, fertile soils with pH 5.8-7.0. Needs good nitrogen levels.",
                "water_needs": "Moderate water requirements, sensitive to drought during silking stage."
            },
            "wheat": {
                "description": "Wheat is a cereal grain and a worldwide staple used to make flour for bread, pasta, and pastry.",
                "growing_conditions": "Cool season crop. Winter wheat planted in fall, spring wheat in spring.",
                "soil_requirements": "Well-drained loamy soils with pH 6.0-7.0. Moderate fertility requirements.",
                "water_needs": "Moderate water requirements, about 450-650 mm during growing season."
            }
        }
        
        # Display information for the predicted crop (if available)
        predicted_crop_lower = predicted_crop.lower()
        if predicted_crop_lower in crop_info:
            info = crop_info[predicted_crop_lower]
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Growing Conditions:** {info['growing_conditions']}")
            st.markdown(f"**Soil Requirements:** {info['soil_requirements']}")
            st.markdown(f"**Water Needs:** {info['water_needs']}")
        else:
            st.info(f"Detailed information for {predicted_crop} is not available in our database.")

if __name__ == "__main__":
    main() 