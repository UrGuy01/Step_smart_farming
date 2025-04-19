import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the model and necessary files
model_path = 'model/random_forest_model.pkl'
scaler_path = 'processed_data/scaler.pkl'
encoder_path = 'processed_data/label_encoder.pkl'
feature_stats_path = 'processed_data/feature_stats.csv'

@st.cache_resource
def load_model():
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)
    feature_stats = pd.read_csv(feature_stats_path)
    return model, scaler, label_encoder, feature_stats

# Function to predict crop
def predict_crop(N, P, K, temperature, humidity, ph, rainfall, model, scaler, label_encoder):
    # Create input array
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # Get the crop name
    crop_name = label_encoder.inverse_transform([prediction])[0]
    
    # Get probability scores
    probabilities = model.predict_proba(input_scaled)[0]
    
    # Get top 3 predictions with probabilities
    top_indices = probabilities.argsort()[-3:][::-1]
    top_crops = label_encoder.inverse_transform(top_indices)
    top_probs = probabilities[top_indices]
    
    return crop_name, list(zip(top_crops, top_probs))

# Function to get feature importance data
def get_feature_importance_data(model):
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    importance_values = model.feature_importances_
    
    # Create a dataframe for the feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    return feature_importance

# Function to plot feature importance (no caching)
def plot_feature_importance(model):
    # Get the cached data
    feature_importance = get_feature_importance_data(model)
    
    # Create a new figure each time
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
    ax.set_title('Feature Importance')
    plt.tight_layout()
    return fig

# Main app
def main():
    st.set_page_config(
        page_title="Crop Recommendation System",
        page_icon="🌱",
        layout="wide"
    )
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error("Model file not found. Please run crop_prediction_model.py first to train the model.")
        return
    
    # Load model and related files
    model, scaler, label_encoder, feature_stats = load_model()
    
    # App title and description
    st.title("🌱 Smart Farming Crop Recommendation System")
    st.write("""
    This application helps farmers determine the optimal crop to plant based on soil and climate conditions.
    Enter your soil and climate parameters below to get a recommendation.
    """)
    
    # Create sidebar for inputs
    st.sidebar.header("Soil and Climate Parameters")
    
    # Add input sliders with appropriate ranges from feature stats
    N = st.sidebar.slider("Nitrogen (N) content in soil", 
                         int(max(0, feature_stats.loc[feature_stats['feature'] == 'N', 'mean'].values[0] - 3 * feature_stats.loc[feature_stats['feature'] == 'N', 'scale'].values[0])),
                         int(feature_stats.loc[feature_stats['feature'] == 'N', 'mean'].values[0] + 3 * feature_stats.loc[feature_stats['feature'] == 'N', 'scale'].values[0]),
                         50)
    
    P = st.sidebar.slider("Phosphorus (P) content in soil", 
                         int(max(0, feature_stats.loc[feature_stats['feature'] == 'P', 'mean'].values[0] - 3 * feature_stats.loc[feature_stats['feature'] == 'P', 'scale'].values[0])),
                         int(feature_stats.loc[feature_stats['feature'] == 'P', 'mean'].values[0] + 3 * feature_stats.loc[feature_stats['feature'] == 'P', 'scale'].values[0]),
                         50)
    
    K = st.sidebar.slider("Potassium (K) content in soil", 
                         int(max(0, feature_stats.loc[feature_stats['feature'] == 'K', 'mean'].values[0] - 3 * feature_stats.loc[feature_stats['feature'] == 'K', 'scale'].values[0])),
                         int(feature_stats.loc[feature_stats['feature'] == 'K', 'mean'].values[0] + 3 * feature_stats.loc[feature_stats['feature'] == 'K', 'scale'].values[0]),
                         50)
    
    temperature = st.sidebar.slider("Temperature (°C)", 
                                  float(max(5, feature_stats.loc[feature_stats['feature'] == 'temperature', 'mean'].values[0] - 3 * feature_stats.loc[feature_stats['feature'] == 'temperature', 'scale'].values[0])),
                                  float(feature_stats.loc[feature_stats['feature'] == 'temperature', 'mean'].values[0] + 3 * feature_stats.loc[feature_stats['feature'] == 'temperature', 'scale'].values[0]),
                                  25.0)
    
    humidity = st.sidebar.slider("Humidity (%)", 
                               float(max(0, feature_stats.loc[feature_stats['feature'] == 'humidity', 'mean'].values[0] - 3 * feature_stats.loc[feature_stats['feature'] == 'humidity', 'scale'].values[0])),
                               float(min(100, feature_stats.loc[feature_stats['feature'] == 'humidity', 'mean'].values[0] + 3 * feature_stats.loc[feature_stats['feature'] == 'humidity', 'scale'].values[0])),
                               70.0)
    
    ph = st.sidebar.slider("pH value", 
                          float(max(0, feature_stats.loc[feature_stats['feature'] == 'ph', 'mean'].values[0] - 3 * feature_stats.loc[feature_stats['feature'] == 'ph', 'scale'].values[0])),
                          float(feature_stats.loc[feature_stats['feature'] == 'ph', 'mean'].values[0] + 3 * feature_stats.loc[feature_stats['feature'] == 'ph', 'scale'].values[0]),
                          6.5)
    
    rainfall = st.sidebar.slider("Rainfall (mm)", 
                               float(max(0, feature_stats.loc[feature_stats['feature'] == 'rainfall', 'mean'].values[0] - 3 * feature_stats.loc[feature_stats['feature'] == 'rainfall', 'scale'].values[0])),
                               float(feature_stats.loc[feature_stats['feature'] == 'rainfall', 'mean'].values[0] + 3 * feature_stats.loc[feature_stats['feature'] == 'rainfall', 'scale'].values[0]),
                               100.0)
    
    st.sidebar.info("Adjust the sliders to match your soil and climate conditions.")
    
    # Create columns for the main content
    col1, col2 = st.columns([2, 1])
    
    # Make prediction when user clicks predict button
    if st.sidebar.button("Predict Recommended Crop"):
        with st.spinner("Predicting..."):
            recommended_crop, top_predictions = predict_crop(
                N, P, K, temperature, humidity, ph, rainfall, model, scaler, label_encoder
            )
        
        # Display results
        with col1:
            st.subheader("Prediction Results")
            st.success(f"The recommended crop for your soil and climate conditions is: **{recommended_crop.upper()}**")
            
            # Display top 3 predictions with probabilities
            st.write("Top Predictions:")
            
            # Create a dataframe for the predictions
            predictions_df = pd.DataFrame(top_predictions, columns=['Crop', 'Probability'])
            predictions_df['Probability'] = predictions_df['Probability'].apply(lambda x: f"{x:.2%}")
            
            # Style the dataframe
            st.dataframe(predictions_df, width=500)
            
            # Display input parameters
            st.subheader("Your Input Parameters")
            input_data = pd.DataFrame({
                'Parameter': ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)', 
                             'Temperature (°C)', 'Humidity (%)', 'pH', 'Rainfall (mm)'],
                'Value': [N, P, K, temperature, humidity, ph, rainfall]
            })
            st.dataframe(input_data, width=500)
        
        with col2:
            # Display feature importance
            st.subheader("Feature Importance")
            fig = plot_feature_importance(model)
            st.pyplot(fig)
    
    # Add informational content
    st.markdown("---")
    
    # About section
    st.header("About this System")
    st.write("""
    This crop recommendation system uses a machine learning model trained on soil and climate data to predict the most suitable crops.
    
    The system considers seven key parameters:
    - **Nitrogen (N)**: Essential for plant growth and protein synthesis
    - **Phosphorus (P)**: Important for root development and energy transfer
    - **Potassium (K)**: Helps in disease resistance and water regulation
    - **Temperature**: Affects plant growth rate and metabolism
    - **Humidity**: Influences transpiration and water uptake
    - **pH**: Affects nutrient availability in soil
    - **Rainfall**: Determines water availability for crops
    
    The model has been trained on a dataset containing information about various crops and their optimal growing conditions.
    """)
    
    # Show raw dataset
    if st.checkbox("Show Dataset Information"):
        try:
            crop_data = pd.read_csv("data/raw/Crop_recommendation.csv")
            st.write("Dataset Sample:", crop_data.head())
            
            # Show crop statistics
            st.write("Crop Statistics:")
            crop_stats = crop_data.groupby('label').mean().reset_index()
            st.dataframe(crop_stats)
        except:
            st.warning("Dataset file not available. Please ensure 'data/raw/Crop_recommendation.csv' exists.")

if __name__ == "__main__":
    main() 