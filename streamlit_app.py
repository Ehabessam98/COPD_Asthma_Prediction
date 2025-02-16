import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("asthma_prediction_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# App title
st.title("COPD & Asthma Prediction")
st.write("Enter patient details to predict the condition.")

# User input fields
age = st.number_input("Age", min_value=1, max_value=120, value=30)
peak_flow = st.number_input("Peak Flow (L/min)", min_value=0, value=250)
medication_inhaler = st.radio("Uses Medication Inhaler?", ("Yes", "No"))

# Convert input to model format
medication_inhaler = 1 if medication_inhaler == "Yes" else 0
input_data = np.array([[peak_flow, age, medication_inhaler]])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)
    result = "COPD" if prediction[0] == 1 else "Asthma"
    st.success(f"Prediction: {result}")
