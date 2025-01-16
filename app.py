import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the trained model
model = joblib.load("obesity_model.pkl")

# Load training column names for alignment
model_training_columns = joblib.load("model_training_columns.pkl")  # Save these during training

# Sidebar for user input
st.sidebar.header("User Input Parameters")

# Function to get user input
def user_input_features():
    Age = st.sidebar.slider("Age", 10, 80, 22)
    Height = st.sidebar.slider("Height (in cm)", 130, 200, 178)
    Weight = st.sidebar.slider("Weight (in kg)", 30, 150, 90)
    family_history = st.sidebar.selectbox("Family History of Obesity", ["Yes", "No"])
    FAVC = st.sidebar.selectbox("Frequent Consumption of High Caloric Food (FAVC)", ["Yes", "No"])
    SMOKE = st.sidebar.selectbox("Smokes?", ["Yes", "No"])
    SCC = st.sidebar.selectbox("Chronic Disease?", ["Yes", "No"])
    FAF = st.sidebar.selectbox("Physical Activity (FAF)", ["Low", "Medium", "High", "Frequent"])
    MTRANS = st.sidebar.selectbox("Mode of Transportation (MTRANS)", ["Public_Transportation", "Bike", "Motorbike", "Walking", "Automobile"])
    CAEC = st.sidebar.selectbox("Eating Habit (CAEC)", ["No", "Sometimes", "Frequently", "Always"])
    CALC = st.sidebar.selectbox("Caloric Intake (CALC)", ["No", "Sometimes", "Frequently", "Always"])
    
    # Return as a dictionary
    data = {
        "Age": Age,
        "Height": Height / 100,  # Convert cm to meters for consistency
        "Weight": Weight,
        "family_history": family_history,
        "FAVC": FAVC,
        "SMOKE": SMOKE,
        "SCC": SCC,
        "FAF": FAF,
        "MTRANS": MTRANS,
        "CAEC": CAEC,
        "CALC": CALC,
    }
    return data

# Function to preprocess user input
def preprocess_user_input(user_input):
    # Convert input to DataFrame
    df_input = pd.DataFrame([user_input])
    
    # Label encode binary columns
    binary_columns = ["family_history", "FAVC", "SMOKE", "SCC"]
    le = LabelEncoder()
    for col in binary_columns:
        df_input[col] = le.fit_transform(df_input[col])
    
    # One-hot encode categorical variables
    df_input = pd.get_dummies(df_input, columns=["FAF", "MTRANS", "CAEC", "CALC"], drop_first=True)
    
    # Align columns with training data
    missing_cols = set(model_training_columns) - set(df_input.columns)
    for col in missing_cols:
        df_input[col] = 0
    df_input = df_input[model_training_columns]  # Reorder columns
    
    return df_input

# Get user input
user_input = user_input_features()
st.subheader("User Input Parameters")
st.write(pd.DataFrame([user_input]))

# Preprocess user input
preprocessed_input = preprocess_user_input(user_input)

# Debugging
st.write("Preprocessed Input Shape:", preprocessed_input.shape)
st.write("Preprocessed Input Columns:", preprocessed_input.columns)

# Make prediction
prediction = model.predict(preprocessed_input)
prediction_proba = model.predict_proba(preprocessed_input)

# Map predictions to class labels
obesity_levels = {
    0: "Insufficient_Weight",
    1: "Normal_Weight",
    2: "Overweight_Level_I",
    3: "Overweight_Level_II",
    4: "Obesity_Type_I",
    5: "Obesity_Type_II",
    6: "Obesity_Type_III",
}

# Display prediction and probabilities
st.subheader("Prediction")
st.write(f"Predicted Obesity Level: {obesity_levels[prediction[0]]}")

st.subheader("Prediction Probability")
st.write(f"Probability of the predicted obesity level: {prediction_proba[0][prediction[0]] * 100:.2f}%")

st.subheader("Class labels and their corresponding index number")
st.write(obesity_levels)
