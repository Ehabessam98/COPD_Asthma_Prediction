import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("copd_asthma_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("COPD & Asthma Prediction")
st.write("Enter patient details to predict the condition.")

# User Input Fields
age = st.number_input("Age", min_value=1, max_value=120, value=60)
peak_flow = st.number_input("Peak Flow (L/min)", min_value=50, max_value=800, value=250)
uses_inhaler = st.radio("Uses Medication Inhaler?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

# Predict Button
if st.button("Predict"):
    input_data = np.array([[age, peak_flow, uses_inhaler]])  # Ensure correct shape
    prediction = model.predict(input_data)[0]  # Get prediction result
    
    if prediction == 1:
        st.error("🔴 High Risk: The model predicts COPD or Asthma.")
    else:
        st.success("🟢 Low Risk: The model predicts no COPD or Asthma.")
