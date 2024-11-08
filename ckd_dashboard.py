import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Title of the dashboard
st.title('Chronic Kidney Disease (CKD) Prediction')

# Load the cleaned dataset (use the path to your dataset)
@st.cache
def load_data():
    data = pd.read_csv("cleaned_ckd_data_winsorized.csv")  # Modify with your actual data path
    return data

# Display the data
st.header("Dataset")
df = load_data()
st.write(df.head())

# Preprocessing
st.header("Preprocessing the Data")
df['classification'] = LabelEncoder().fit_transform(df['classification'])

# Fill missing values if any
df.fillna(df.median(), inplace=True)

# Display a summary of missing values and data types
st.write("Missing Values:")
st.write(df.isnull().sum())

# Create a sidebar for model inputs
st.sidebar.header("Model Inputs")

age = st.sidebar.slider('Age', min_value=2, max_value=90, value=50)
bp = st.sidebar.slider('Blood Pressure', min_value=50, max_value=180, value=80)
sg = st.sidebar.slider('Specific Gravity', min_value=1.005, max_value=1.025, value=1.020)
al = st.sidebar.slider('Albumin', min_value=0, max_value=5, value=0)
su = st.sidebar.slider('Sugar', min_value=0, max_value=5, value=0)

# Collecting other necessary features similarly
# ...

# Model training and prediction
st.header("Model Training and Prediction")

# Prepare features and labels for the model
X = df.drop(columns=["classification", "id"])  # Drop target and unnecessary columns
y = df["classification"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier (or other model)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display results
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# User input prediction
input_data = np.array([age, bp, sg, al, su]).reshape(1, -1)
prediction = model.predict(input_data)
prediction_label = 'CKD' if prediction == 1 else 'Not CKD'

st.sidebar.write(f"Prediction: {prediction_label}")

# Displaying relevant graphs and insights
st.header("Key Insights and Visualizations")
st.subheader("Age vs Blood Pressure")
st.line_chart(df[['age', 'bp']].dropna())



