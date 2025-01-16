import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model
try:
    model = joblib.load("obesity_model.pkl")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Define label encoding mappings for consistent processing
label_to_index = {
    "Insufficient_Weight": 0,
    "Normal_Weight": 1,
    "Obesity_Type_I": 2,
    "Obesity_Type_II": 3,
    "Obesity_Type_III": 4,
    "Overweight_Level_I": 5,
    "Overweight_Level_II": 6,
}
index_to_label = {v: k for k, v in label_to_index.items()}

# Sidebar for user input
st.sidebar.header("User Input Parameters")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.slider("Age", 10, 80, 30)
    height = st.sidebar.slider("Height (in cm)", 130, 200, 170)
    weight = st.sidebar.slider("Weight (in kg)", 30, 150, 70)
    family_history = st.sidebar.selectbox("Family History of Obesity", ["Yes", "No"])
    favc = st.sidebar.selectbox("Frequent Consumption of High Caloric Food (FAVC)", ["Yes", "No"])
    smoke = st.sidebar.selectbox("Smokes?", ["Yes", "No"])
    scc = st.sidebar.selectbox("Monitor Calories (SCC)?", ["Yes", "No"])
    faf = st.sidebar.selectbox("Physical Activity (FAF)", ["Low", "Medium", "High"])
    mtrans = st.sidebar.selectbox("Mode of Transportation (MTRANS)", ["Walking", "Public_Transportation", "Automobile", "Bike", "Motorbike"])
    caec = st.sidebar.selectbox("Eating Habit (CAEC)", ["No", "Sometimes", "Frequently", "Always"])
    calc = st.sidebar.selectbox("Caloric Intake (CALC)", ["No", "Sometimes", "Frequently", "Always"])
    
    data = {
        "Gender": gender,
        "Age": age,
        "Height": height / 100,  # Convert to meters
        "Weight": weight,
        "family_history": family_history,
        "FAVC": favc,
        "SMOKE": smoke,
        "SCC": scc,
        "FAF": faf,
        "MTRANS": mtrans,
        "CAEC": caec,
        "CALC": calc,
    }
    features = pd.DataFrame(data, index=[0])
    return features

user_input = user_input_features()

# Show the user input
st.subheader("User Input Parameters")
st.write(user_input)

# Preprocess the user input
def preprocess_user_input(df_input):
    df_input = df_input.copy()
    
    # Encoding for categorical variables
    label_encodings = {
        "Gender": {"Male": 0, "Female": 1},
        "family_history": {"Yes": 1, "No": 0},
        "FAVC": {"Yes": 1, "No": 0},
        "SMOKE": {"Yes": 1, "No": 0},
        "SCC": {"Yes": 1, "No": 0},
        "FAF": {"Low": 0, "Medium": 1, "High": 2},
        "MTRANS": {"Walking": 0, "Public_Transportation": 1, "Automobile": 2, "Bike": 3, "Motorbike": 4},
        "CAEC": {"No": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
        "CALC": {"No": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
    }
    
    for col, mapping in label_encodings.items():
        if col in df_input.columns:
            df_input[col] = df_input[col].map(mapping)
    return df_input

preprocessed_input = preprocess_user_input(user_input)

# Prediction
try:
    prediction_index = model.predict(preprocessed_input)[0]
    prediction_label = index_to_label[prediction_index]
    prediction_proba = model.predict_proba(preprocessed_input)
    predicted_class_proba = prediction_proba[0][prediction_index]
except Exception as e:
    st.error(f"Prediction Error: {e}")
    st.stop()

# Display prediction
st.subheader("Prediction")
st.write(f"Predicted Obesity Level: {prediction_label}")

# Display prediction probability
st.subheader("Prediction Probability")
st.write(f"Probability of the predicted obesity level: {predicted_class_proba * 100:.2f}%")

# Display label-to-index mapping for debugging
st.subheader("Class labels and their corresponding index number")
st.write(index_to_label)
