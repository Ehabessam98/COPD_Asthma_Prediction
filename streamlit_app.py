import streamlit as st
import numpy as np
import pickle

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open("asthma_prediction_model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Error: Model file not found! Please upload `asthma_prediction_model.pkl` to the project folder.")
        return None

model = load_model()

# Title and input form
st.title("COPD & Asthma Prediction")
st.write("Enter patient details to predict the condition.")

# User Inputs
age = st.number_input("Age", min_value=0, max_value=120, value=60)
peak_flow = st.number_input("Peak Flow (L/min)", min_value=50, max_value=800, value=250)
medication_inhaler = st.radio("Uses Medication Inhaler?", ["Yes", "No"])

# Convert user input to numerical format
medication_inhaler = 1 if medication_inhaler == "Yes" else 0

# Prediction button
if st.button("Predict"):
    if model is not None:
        input_data = np.array([[age, peak_flow, medication_inhaler]])  # Format input for the model
        try:
            prediction = model.predict(input_data)  # Make prediction
            result = "COPD Detected" if prediction[0] == 1 else "Asthma Detected"
            st.success(f"Prediction: {result}")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.warning("Model is not loaded. Please check the file.")
