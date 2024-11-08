import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model and label encoders
model = joblib.load('ckd_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Define a function to preprocess input data
def preprocess_input(input_data):
    # Here, you should apply the same preprocessing you used during training
    # e.g., encoding categorical variables, scaling, etc.
    for column, le in label_encoders.items():
        input_data[column] = le.transform([input_data[column]])[0]  # Example of encoding categorical columns
    return input_data

# Streamlit app
st.title('Chronic Kidney Disease (CKD) Prediction')

# Input fields for the user to provide data
age = st.number_input('Age', min_value=1, max_value=120, value=30)
bp = st.number_input('Blood Pressure')
sg = st.number_input('Specific Gravity', min_value=1.005, max_value=1.050, value=1.020)
al = st.number_input('Albumin')
su = st.number_input('Sugar')
rbc = st.selectbox('Red Blood Cells', ['normal', 'abnormal'])
pc = st.selectbox('Pus Cells', ['normal', 'abnormal'])
pcc = st.selectbox('Pus Cell Clumps', ['present', 'notpresent'])
ba = st.selectbox('Bacteria', ['present', 'notpresent'])
bgr = st.number_input('Blood Glucose Random', value=100)
bu = st.number_input('Blood Urea')
sc = st.number_input('Serum Creatinine')
sod = st.number_input('Sodium')
pot = st.number_input('Potassium')
hemo = st.number_input('Hemoglobin')
pcv = st.number_input('Packed Cell Volume')
wc = st.number_input('White Blood Cell Count')
rc = st.number_input('Red Blood Cell Count')
htn = st.selectbox('Hypertension', ['yes', 'no'])
dm = st.selectbox('Diabetes Mellitus', ['yes', 'no'])
cad = st.selectbox('Coronary Artery Disease', ['yes', 'no'])
appet = st.selectbox('Appetite', ['good', 'poor'])
pe = st.selectbox('Polydipsia', ['yes', 'no'])
ane = st.selectbox('Anemia', ['yes', 'no'])

# Store input data in a dictionary
input_data = {
    'age': age,
    'bp': bp,
    'sg': sg,
    'al': al,
    'su': su,
    'rbc': rbc,
    'pc': pc,
    'pcc': pcc,
    'ba': ba,
    'bgr': bgr,
    'bu': bu,
    'sc': sc,
    'sod': sod,
    'pot': pot,
    'hemo': hemo,
    'pcv': pcv,
    'wc': wc,
    'rc': rc,
    'htn': htn,
    'dm': dm,
    'cad': cad,
    'appet': appet,
    'pe': pe,
    'ane': ane
}

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Preprocess the input data (if necessary)
processed_input = preprocess_input(input_df)

# Make prediction
prediction = model.predict(processed_input)

# Display the prediction result
if prediction == 1:
    st.write('Prediction: CKD Detected')
else:
    st.write('Prediction: No CKD Detected')

