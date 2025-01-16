import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and preprocessing objects
model = joblib.load("obesity_model.pkl")
scaler = joblib.load("scaler.pkl")
training_columns = joblib.load("training_columns.pkl")  # Columns used during training

# Mapping of labels to their corresponding index
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

st.title("Obesity Prediction App")
st.write("This app predicts the Obesity Level based on your inputs!")

# User input features
st.sidebar.header("User Input Parameters")
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
age = st.sidebar.slider("Age", 10, 80, 25)
height = st.sidebar.slider("Height (in cm)", 130, 200, 170)
weight = st.sidebar.slider("Weight (in kg)", 30, 150, 70)
family_history = st.sidebar.selectbox("Family History of Obesity", options=["Yes", "No"])
favc = st.sidebar.selectbox("Frequent Consumption of High Caloric Food (FAVC)", options=["Yes", "No"])
smoke = st.sidebar.selectbox("Smokes?", options=["Yes", "No"])
scc = st.sidebar.selectbox("Calories Monitoring (SCC)", options=["Yes", "No"])
faf = st.sidebar.selectbox("Physical Activity (FAF)", options=["None", "Low", "Medium", "High"])
mtrans = st.sidebar.selectbox("Mode of Transportation (MTRANS)", options=["Walking", "Bike", "Motorbike", "Public_Transportation", "Automobile"])
caec = st.sidebar.selectbox("Eating Habit (CAEC)", options=["No", "Sometimes", "Frequently", "Always"])
calc = st.sidebar.selectbox("Caloric Intake (CALC)", options=["No", "Sometimes", "Frequently", "Always"])

# Store user inputs into a DataFrame
user_input = {
    "Gender": gender,
    "Age": age,
    "Height": height / 100,  # Convert cm to meters
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
df_input = pd.DataFrame([user_input])

st.write("### User Input Parameters")
st.write(df_input)

# Preprocess the user input
def preprocess_user_input(df):
    df = df.copy()

    # Encode categorical variables
    categorical_features = ["Gender", "family_history", "FAVC", "SMOKE", "SCC", "FAF", "MTRANS", "CAEC", "CALC"]
    for col in categorical_features:
        df[col] = df[col].astype("category")
        df[col] = df[col].cat.codes

    # Scale numerical features
    numerical_features = ["Age", "Height", "Weight"]
    df[numerical_features] = scaler.transform(df[numerical_features])

    # Align columns with the training dataset
    df = pd.get_dummies(df)
    df = df.reindex(columns=training_columns, fill_value=0)

    return df

# Preprocess the input
try:
    preprocessed_input = preprocess_user_input(df_input)
except Exception as e:
    st.error(f"Error in preprocessing: {e}")
    st.stop()

# Debugging: Print processed input
st.write("### Preprocessed Input Data")
st.write(preprocessed_input)

# Predict using the model
try:
    prediction_index = model.predict(preprocessed_input)[0]
    prediction_proba = model.predict_proba(preprocessed_input)
    prediction_label = index_to_label[prediction_index]
except Exception as e:
    st.error(f"Prediction error: {e}")
    st.stop()

# Display the prediction
st.subheader("Prediction")
st.write(f"Predicted Obesity Level: {prediction_label}")

# Display the prediction probability
st.subheader("Prediction Probability")
try:
    predicted_class_proba = prediction_proba[0][label_to_index[prediction_label]]
    st.write(f"Probability of the predicted obesity level: {predicted_class_proba * 100:.2f}%")
except Exception as e:
    st.error(f"Probability calculation error: {e}")

# Display class labels for reference
st.subheader("Class labels and their corresponding index number")
st.write(index_to_label)
