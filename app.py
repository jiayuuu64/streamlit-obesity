import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
@st.cache_data
def load_data():
    # Replace with the path to your dataset
    df = pd.read_csv("obesity.csv")
    return df

data = load_data()

# Preprocessing the dataset
def preprocess_dataset(df):
    df = df.copy()
    
    # Encoding categorical columns
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
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    # Encoding the target variable
    target_encoder = LabelEncoder()
    df["Obesity"] = target_encoder.fit_transform(df["Obesity"])
    
    return df, target_encoder

processed_data, target_encoder = preprocess_dataset(data)

# Splitting the dataset
X = processed_data.drop("Obesity", axis=1)
y = processed_data["Obesity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
@st.cache_resource
def train_model():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

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
    prediction_label = target_encoder.inverse_transform([prediction_index])[0]
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

# Display the dataset and model training details
st.subheader("Dataset Overview")
st.write(data)
