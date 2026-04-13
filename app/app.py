import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("models/heart_pipeline.pkl", "rb"))

st.title("❤️ Heart Disease Prediction")

st.write("Enter patient details:")

# Inputs
age = st.slider("Age", 20, 100, 50)

sex = st.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex == "Male" else 0

chest_pain_type = st.selectbox("Chest Pain Type", ['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'])

resting_blood_pressure = st.number_input("Resting Blood Pressure", 80, 200, 120)

cholestoral = st.number_input("Cholesterol", 100, 600, 200)

fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])
fasting_blood_sugar = 1 if fasting_blood_sugar == "No" else 0

rest_ecg = st.selectbox("Rest ECG", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])

Max_heart_rate = st.number_input("Max Heart Rate", 60, 220, 150)

exercise_induced_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
exercise_induced_angina = 1 if exercise_induced_angina == "Yes" else 0

oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)

slope_map = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

slope_display = st.selectbox(
    "Slope",
    list(slope_map.keys())
)

slope = slope_map[slope_display]

vessel_map = {
    "Zero": 0,
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4
}

vessel_display = st.selectbox("Number of Major Vessels", list(vessel_map.keys()))
vessels = vessel_map[vessel_display]

thalassemia = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

# Create dataframe
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'chest_pain_type': [chest_pain_type],
    'resting_blood_pressure': [resting_blood_pressure],
    'cholestoral': [cholestoral],
    'fasting_blood_sugar': [fasting_blood_sugar],
    'rest_ecg': [rest_ecg],
    'Max_heart_rate': [Max_heart_rate],
    'exercise_induced_angina': [exercise_induced_angina],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'vessels_colored_by_flourosopy': [vessels],
    'thalassemia': [thalassemia]
})

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"High Risk of Heart Disease ({prob:.2f})")
    else:
        st.success(f"Low Risk of Heart Disease ({prob:.2f})")

