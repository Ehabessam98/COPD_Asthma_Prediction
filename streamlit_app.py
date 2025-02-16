import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("copd_asthma_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("COPD - Asthma Prediction")
st.write("Enter patient details to predict the condition.")

# User Input Fields
age = st.number_input("Age", min_value=1, max_value=52, value=30)
peak_flow = st.number_input("Peak Flow (L/min)", min_value=50, max_value=800, value=250)
smoking = st.radio("Do you smoke?", ["Yes", "No"])
cough = st.radio("Do you have a persistent cough?", ["Yes", "No"])

# Predict Button
if st.button("Predict"):
    # Convert inputs into numerical values for the model
    smoking_value = 1 if smoking == "Yes" else 0
    cough_value = 1 if cough == "Yes" else 0

    # Prepare input for prediction
    test_input = np.array([[age, peak_flow, smoking_value, cough_value]])

    # Make prediction
    prediction = model.predict(test_input)[0]  # Get prediction result
    
    # Display results
    if prediction == 1:
        st.error("🔴 High Risk: The model predicts COPD or Asthma.")
    else:
        st.success("🟢 Low Risk: The model predicts no COPD or Asthma.")

# Footer
st.markdown(
    """
    <div style="position: fixed; bottom: 10px; width: 100%; text-align: center; font-size: 14px; color: gray;">
        Developed by <b>Ehab Essam</b>
    </div>
    """,
    unsafe_allow_html=True
)
