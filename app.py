import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page layout
st.set_page_config(page_title="Used Car Price Predictor", layout="centered")
st.title("🚗 Used Car Price Predictor")
st.write("Enter the vehicle details below to estimate its market value (in Lakhs).")

# 1. Load the model and columns safely
@st.cache_resource
def load_assets():
    # Get the exact directory where app.py lives
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(BASE_DIR, 'car_price_model.pkl')
    columns_path = os.path.join(BASE_DIR, 'model_columns.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(columns_path, 'rb') as f:
        columns = pickle.load(f)
    return model, columns

try:
    model, model_columns = load_assets()
except FileNotFoundError:
    st.error("Model files not found! Please run the export code in your notebook first.")
    st.stop()

# 2. Re-create the categorical options directly from your pipeline rules
brands = ['Maruti', 'Mahindra', 'Volkswagen', 'Hyundai', 'Honda', 'Toyota', 'BMW', 'Mercedes', 'Jeep'] 
states = ['Tripura', 'Maharashtra', 'Haryana', 'Rajasthan', 'Uttar Pradesh', 'Gujarat', 'Delhi', 'Punjab', 'Karnataka', 'Tamil Nadu'] 
fuels = ['Diesel', 'Petrol', 'CNG', 'Electric', 'Hybrid']

# 3. Create User Inputs UI
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Select Car Brand", sorted(brands))
    fuel_type = st.selectbox("Select Fuel Type", fuels)
    transmission = st.selectbox("Transmission Type", ["Manual", "Auto"])
    ownership = st.slider("Ownership History", min_value=1, max_value=4, step=1, help="Values 4+ are grouped as 4")

with col2:
    car_age = st.number_input("Car Age (Years)", min_value=0, max_value=30, value=5)
    km_driven = st.number_input("Kilometers Driven", min_value=100, max_value=300000, value=50000, step=1000)
    state = st.selectbox("State of Registration", sorted(states))

# 4. Predict button logic
if st.button("Predict Market Price", type="primary"):
    
    # Create a blank base row matching the exact structure your model expects (all zeros)
    input_data = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # Fill numerical columns
    input_data['KM Driven'] = km_driven
    input_data['Car Age'] = car_age
    input_data['Ownership'] = ownership
    input_data['Transmission Type'] = 1 if transmission == "Auto" else 0
    
    # Activate the matching One-Hot encoded flags if they exist in our columns list
    fuel_col = f"Fuel Type_{fuel_type}"
    state_col = f"State_{state}"
    brand_col = f"Brand_{brand}"
    
    if fuel_col in input_data.columns:
        input_data[fuel_col] = 1
    if state_col in input_data.columns:
        input_data[state_col] = 1
    if brand_col in input_data.columns:
        input_data[brand_col] = 1
        
    # Make prediction
    predicted_price = model.predict(input_data)[0]
    
    # Display Result
    st.success(f"### Estimated Price: ₹ {round(predicted_price, 2)} Lakhs")