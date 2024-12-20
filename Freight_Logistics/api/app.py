import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Ensure set_page_config is the first command
st.set_page_config(page_title="Shipment Delay Predictor", layout="wide")

# Load the trained model
MODEL_PATH = '../models/scaler.pkl'

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Define the Streamlit app
st.title("📦 Shipment Delay Predictor")
st.markdown("""
This tool predicts whether a shipment will be delayed or delivered on time based on the provided details. Fill in the form below to get a prediction.
""")

# Input form
st.sidebar.header("Enter Shipment Details")
distance = st.sidebar.number_input("Distance (km)", min_value=0, step=1, value=100)
planned_delivery_days = st.sidebar.number_input("Planned Delivery Days", min_value=1, step=1, value=2)
actual_delivery_days = st.sidebar.number_input("Actual Delivery Days", min_value=1, step=1, value=2)
is_long_distance = st.sidebar.radio("Is it a long-distance shipment?", options=[0, 1], index=0)
is_bad_weather = st.sidebar.radio("Are the weather conditions bad?", options=[0, 1], index=0)
is_heavy_traffic = st.sidebar.radio("Are there heavy traffic conditions?", options=[0, 1], index=0)

# Predict button
if st.sidebar.button("Predict Delay"):
    # Prepare the input for the model
    input_data = np.array([[distance, planned_delivery_days, actual_delivery_days, 
                            is_long_distance, is_bad_weather, is_heavy_traffic]])
    
    # Prediction
    prediction = model.predict(input_data)[0]
    delay_probability = model.predict_proba(input_data)[0][1]
    
    # Display the result
    st.header("Prediction Results")
    if prediction == 1:
        st.error(f"🚨 The shipment is likely to be delayed with a probability of {delay_probability:.2%}.")
    else:
        st.success(f"✅ The shipment is expected to be on time with a probability of {1 - delay_probability:.2%}.")
else:
    st.info("Enter shipment details in the sidebar and click 'Predict Delay'.")

# Footer
st.markdown("""
---
**Note:** This application uses a machine learning model trained on historical shipment data to make predictions. For best results, provide accurate input details.
""")
