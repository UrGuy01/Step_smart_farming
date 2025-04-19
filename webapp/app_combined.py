import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the model and necessary files
combined_model_path = 'model/random_forest_combined.pkl'
extended_model_path = 'model/random_forest_extended.pkl'
combined_scaler_path = 'processed_data/scaler_combined.pkl'
label_encoder_path = 'processed_data/label_encoder_combined.pkl'
feature_stats_path = 'processed_data/feature_stats_combined.csv'
extended_preprocessor_path = 'processed_data/extended_preprocessor.pkl'
extended_features_path = 'processed_data/extended_feature_names.csv'

@st.cache_resource
def load_models():
    # Check if files exist
    if not os.path.exists(combined_model_path):
        return None, None, None, None, None, None
    
    combined_model = joblib.load(combined_model_path)
    combined_scaler = joblib.load(combined_scaler_path)
    label_encoder = joblib.load(label_encoder_path)
    feature_stats = pd.read_csv(feature_stats_path)
    
    # Check if extended model exists
    extended_model = None
    extended_preprocessor = None
    if os.path.exists(extended_model_path) and os.path.exists(extended_preprocessor_path):
        extended_model = joblib.load(extended_model_path)
        extended_preprocessor = joblib.load(extended_preprocessor_path)
    
    return combined_model, combined_scaler, label_encoder, feature_stats, extended_model, extended_preprocessor

# Function to get feature importance data
@st.cache_data
def get_feature_importance_data(model, feature_names):
    importance_values = model.feature_importances_
    
    # Create a dataframe for the feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    return feature_importance

# Function to plot feature importance
def plot_feature_importance(model, feature_names):
    # Get the cached data
    feature_importance = get_feature_importance_data(model, feature_names)
    
    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), ax=ax)
    ax.set_title('Top 10 Feature Importance')
    plt.tight_layout()
    return fig

# Function to predict crop using the combined model
def predict_crop_combined(N, P, K, temperature, humidity, ph, rainfall, model, scaler, label_encoder):
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

# Function to predict crop using the extended model
def predict_crop_extended(N, P, K, temperature, humidity, ph, rainfall, Soilcolor, Zn, S, 
                         WD10M, GWETTOP, CLOUD_AMT, WS2M_RANGE, PS, model, preprocessor, label_encoder):
    # Create input dataframe with all features
    input_data = pd.DataFrame({
        'N': [N], 'P': [P], 'K': [K], 'temperature': [temperature], 
        'humidity': [humidity], 'ph': [ph], 'rainfall': [rainfall], 
        'Soilcolor': [Soilcolor], 'Zn': [Zn], 'S': [S], 
        'WD10M': [WD10M], 'GWETTOP': [GWETTOP], 'CLOUD_AMT': [CLOUD_AMT], 
        'WS2M_RANGE': [WS2M_RANGE], 'PS': [PS]
    })
    
    # Apply preprocessing
    input_processed = preprocessor.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_processed)[0]
    
    # Get the crop name
    crop_name = label_encoder.inverse_transform([prediction])[0]
    
    # Get probability scores
    probabilities = model.predict_proba(input_processed)[0]
    
    # Get top 3 predictions with probabilities
    top_indices = probabilities.argsort()[-3:][::-1]
    top_crops = label_encoder.inverse_transform(top_indices)
    top_probs = probabilities[top_indices]
    
    return crop_name, list(zip(top_crops, top_probs))

# Main app
def main():
    st.set_page_config(
        page_title="Enhanced Crop Recommendation System",
        page_icon="🌾",
        layout="wide"
    )
    
    # Load models and related files
    combined_model, combined_scaler, label_encoder, feature_stats, extended_model, extended_preprocessor = load_models()
    
    if combined_model is None:
        st.error("Model files not found. Please run the preprocessing and model training scripts first.")
        return
    
    # Get basic feature names
    basic_features = feature_stats['feature'].tolist()
    
    # Load extended feature names if available
    extended_features = []
    if os.path.exists(extended_features_path):
        extended_features = pd.read_csv(extended_features_path)['feature_name'].tolist()
    
    # App title and description
    st.title("🌾 Enhanced Smart Farming Crop Recommendation System")
    st.write("""
    This application helps farmers determine the optimal crop to plant based on soil properties and climate conditions.
    Enter your soil and climate parameters below to get a recommendation.
    """)
    
    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    model_option = st.sidebar.radio(
        "Choose prediction model:",
        ["Basic Model (7 features)", "Extended Model (all features)"] if extended_model else ["Basic Model (7 features)"]
    )
    
    using_extended_model = model_option == "Extended Model (all features)" and extended_model is not None
    
    # Input parameters section
    st.sidebar.header("Soil and Climate Parameters")
    
    # Basic parameters (needed for both models)
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
    
    # Extended parameters section (only shown if using extended model)
    extended_params = {}
    if using_extended_model:
        st.sidebar.markdown("---")
        st.sidebar.header("Additional Parameters")
        
        # Soil color (categorical)
        # Get all possible soil colors from the dataset
        soil_colors_list = ['brown', 'red', 'gray', 'black', 'Yellowish brown', 'Reddish brown']
        extended_params['Soilcolor'] = st.sidebar.selectbox("Soil Color", soil_colors_list)
        
        # Additional numeric parameters
        extended_params['Zn'] = st.sidebar.slider("Zinc (Zn) content", 0.0, 10.0, 2.0)
        extended_params['S'] = st.sidebar.slider("Sulfur (S) content", 0.0, 20.0, 10.0)
        extended_params['WD10M'] = st.sidebar.slider("Wind Direction (WD10M)", 0.0, 360.0, 180.0)
        extended_params['GWETTOP'] = st.sidebar.slider("Top Soil Layer Wetness (GWETTOP)", 0.0, 1.0, 0.7)
        extended_params['CLOUD_AMT'] = st.sidebar.slider("Cloud Amount (%)", 0.0, 100.0, 50.0)
        extended_params['WS2M_RANGE'] = st.sidebar.slider("Wind Speed Range (WS2M_RANGE)", 0.0, 10.0, 5.0)
        extended_params['PS'] = st.sidebar.slider("Surface Pressure (PS)", 70.0, 85.0, 77.0)
    
    st.sidebar.info("Adjust the sliders to match your soil and climate conditions.")
    
    # Create main content columns
    col1, col2 = st.columns([2, 1])
    
    # Make prediction when user clicks the button
    if st.sidebar.button("Predict Recommended Crop"):
        with st.spinner("Predicting..."):
            if using_extended_model:
                recommended_crop, top_predictions = predict_crop_extended(
                    N, P, K, temperature, humidity, ph, rainfall,
                    extended_params['Soilcolor'], extended_params['Zn'], extended_params['S'],
                    extended_params['WD10M'], extended_params['GWETTOP'], extended_params['CLOUD_AMT'],
                    extended_params['WS2M_RANGE'], extended_params['PS'],
                    extended_model, extended_preprocessor, label_encoder
                )
            else:
                recommended_crop, top_predictions = predict_crop_combined(
                    N, P, K, temperature, humidity, ph, rainfall,
                    combined_model, combined_scaler, label_encoder
                )
        
        # Display results
        with col1:
            st.subheader("Prediction Results")
            st.success(f"The recommended crop for your conditions is: **{recommended_crop.upper()}**")
            
            # Display top 3 predictions with probabilities
            st.write("Top Predictions:")
            
            # Create a dataframe for the predictions
            predictions_df = pd.DataFrame(top_predictions, columns=['Crop', 'Probability'])
            predictions_df['Probability'] = predictions_df['Probability'].apply(lambda x: f"{x:.2%}")
            
            # Style the dataframe
            st.dataframe(predictions_df, width=500)
            
            # Display input parameters
            st.subheader("Your Input Parameters")
            
            # Create a dataframe for basic parameters
            input_data = pd.DataFrame({
                'Parameter': ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)', 
                             'Temperature (°C)', 'Humidity (%)', 'pH', 'Rainfall (mm)'],
                'Value': [N, P, K, temperature, humidity, ph, rainfall]
            })
            
            # If using extended model, add the additional parameters
            if using_extended_model:
                ext_params_df = pd.DataFrame({
                    'Parameter': ['Soil Color', 'Zinc (Zn)', 'Sulfur (S)', 'Wind Direction', 
                                 'Soil Wetness', 'Cloud Amount', 'Wind Speed Range', 'Surface Pressure'],
                    'Value': [extended_params['Soilcolor'], extended_params['Zn'], extended_params['S'],
                             extended_params['WD10M'], extended_params['GWETTOP'], extended_params['CLOUD_AMT'],
                             extended_params['WS2M_RANGE'], extended_params['PS']]
                })
                input_data = pd.concat([input_data, ext_params_df], ignore_index=True)
            
            st.dataframe(input_data, width=500)
        
        with col2:
            # Display feature importance
            st.subheader("Feature Importance")
            if using_extended_model:
                fig = plot_feature_importance(extended_model, extended_features)
            else:
                fig = plot_feature_importance(combined_model, basic_features)
            st.pyplot(fig)
    
    # Add informational content
    st.markdown("---")
    
    # About section
    st.header("About this Enhanced System")
    
    st.write("""
    This enhanced crop recommendation system uses machine learning models trained on comprehensive soil and climate data 
    to predict the most suitable crops for your specific conditions.
    
    The system offers two prediction models:
    
    1. **Basic Model**: Uses 7 essential parameters that are common across different datasets
       - Nitrogen (N), Phosphorus (P), Potassium (K): Essential soil nutrients
       - Temperature: Average temperature in degrees Celsius
       - Humidity: Relative humidity percentage
       - pH: Soil acidity/alkalinity
       - Rainfall: Precipitation in millimeters
       
    2. **Extended Model**: Includes additional soil properties and detailed weather parameters
       - Soil Color: Visual characteristic that can indicate soil composition
       - Zinc (Zn) and Sulfur (S): Additional micronutrients
       - Seasonal weather data: More detailed climate information
       - Additional environmental factors: Wind, pressure, cloud cover, etc.
    
    Both models have been trained on agricultural datasets to provide accurate crop recommendations
    based on the conditions you specify.
    """)
    
    # Show combined dataset information
    if st.checkbox("Show Combined Dataset Information"):
        try:
            combined_data = pd.read_csv("processed_data/preprocessed_combined_crop_data.csv")
            
            # Create tabs for different dataset views
            tab1, tab2, tab3 = st.tabs(["Dataset Sample", "Crop Distribution", "Feature Statistics"])
            
            with tab1:
                st.write("Dataset Sample:", combined_data.head())
            
            with tab2:
                crop_counts = combined_data['crop'].value_counts().reset_index()
                crop_counts.columns = ['Crop', 'Count']
                
                # Create a horizontal bar chart
                fig, ax = plt.subplots(figsize=(10, 8))
                chart = sns.barplot(x='Count', y='Crop', data=crop_counts, ax=ax)
                plt.title('Number of Samples per Crop Type')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Also show as a table
                st.dataframe(crop_counts, width=400)
            
            with tab3:
                # Show statistics for numeric features
                numeric_cols = combined_data.select_dtypes(include=[np.number]).columns
                feature_stats = combined_data[numeric_cols].describe().T.reset_index()
                feature_stats.columns = ['Feature', 'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
                st.dataframe(feature_stats, width=800)
        except Exception as e:
            st.warning(f"Could not load dataset information: {e}")

if __name__ == "__main__":
    main() 