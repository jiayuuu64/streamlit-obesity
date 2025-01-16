import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

st.title("Obesity Prediction App")
st.write("This app predicts the Obesity Level based on your inputs!")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("Obesity prediction.csv")
    label_columns = ['Gender', 'family_history', 'FAVC', 'SMOKE', 'SCC']
    le = LabelEncoder()
    for col in label_columns:
        df[col] = le.fit_transform(df[col])
    df = pd.get_dummies(df, columns=['FAF', 'MTRANS', 'CAEC', 'CALC'], drop_first=True)
    return df

data = load_data()
X = data.drop(columns=['Obesity'])
y = data['Obesity']

# Train model
@st.cache_resource
def train_model(X, y):
    model = DecisionTreeClassifier(criterion='entropy', random_state=42)
    model.fit(X, y)
    return model

model = train_model(X, y)

# Obesity level mappings
obesity_levels = {
    "Insufficient_Weight": "Insufficient Weight",
    "Normal_Weight": "Normal Weight",
    "Obesity_Type_I": "Obesity Type I",
    "Obesity_Type_II": "Obesity Type II",
    "Obesity_Type_III": "Obesity Type III",
    "Overweight_Level_I": "Overweight Level I",
    "Overweight_Level_II": "Overweight Level II"
}

# Sidebar for user inputs
st.sidebar.header("User Input Parameters")

def user_input_features():
    Gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    Age = st.sidebar.slider("Age", 10, 80, 22)
    Height = st.sidebar.slider("Height (in cm)", 130, 200, 178)
    Weight = st.sidebar.slider("Weight (in kg)", 30, 150, 90)
    family_history = st.sidebar.selectbox("Family History of Obesity", ("Yes", "No"))
    FAVC = st.sidebar.selectbox("Frequent Consumption of High Caloric Food (FAVC)", ("Yes", "No"))
    SMOKE = st.sidebar.selectbox("Smokes?", ("Yes", "No"))
    SCC = st.sidebar.selectbox("Chronic Disease?", ("Yes", "No"))
    FAF = st.sidebar.selectbox("Physical Activity (FAF)", ["Low", "Medium", "High"])
    MTRANS = st.sidebar.selectbox("Mode of Transportation (MTRANS)", ["Walking", "Bike", "Motorbike", "Public_Transportation"])
    CAEC = st.sidebar.selectbox("Eating Habit (CAEC)", ["no", "Sometimes", "Frequently", "Always"])
    CALC = st.sidebar.selectbox("Caloric Intake (CALC)", ["no", "Sometimes", "Frequently", "Always"])

    data = {
        "Gender": Gender,
        "Age": Age,
        "Height": Height / 100,  # Convert cm to meters
        "Weight": Weight,
        "family_history": family_history,
        "FAVC": FAVC,
        "SMOKE": SMOKE,
        "SCC": SCC,
        "FAF": FAF,
        "MTRANS": MTRANS,
        "CAEC": CAEC,
        "CALC": CALC
    }
    return pd.DataFrame(data, index=[0])

user_input = user_input_features()

# Display user input
st.subheader("User Input Parameters")
st.write(user_input)

# Preprocess user input
def preprocess_user_input(input_df):
    input_df = input_df.copy()
    le = LabelEncoder()
    input_df["Gender"] = le.fit_transform(input_df["Gender"])
    input_df["family_history"] = le.fit_transform(input_df["family_history"])
    input_df["FAVC"] = le.fit_transform(input_df["FAVC"])
    input_df["SMOKE"] = le.fit_transform(input_df["SMOKE"])
    input_df["SCC"] = le.fit_transform(input_df["SCC"])
    input_df = pd.get_dummies(input_df, columns=['FAF', 'MTRANS', 'CAEC', 'CALC'], drop_first=True)

    # Align input_df with training data
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    return input_df

preprocessed_input = preprocess_user_input(user_input)

# Make prediction
st.subheader("Prediction")
prediction_label = model.predict(preprocessed_input)[0]  # Directly get predicted label
st.write(f"Predicted Obesity Level: {prediction_label}")

