import streamlit as st
import pandas as pd
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# App Header
st.set_page_config(page_title="Obesity Prediction App", layout="wide")
st.title("Obesity Prediction App 🎯")
st.markdown("""
This app predicts **obesity levels** based on your health and lifestyle inputs.  
Enter your details in the sidebar to get predictions below.  
""")

st.markdown("---")

# Sidebar
st.sidebar.header("User Input Parameters")
st.sidebar.markdown("""
Please enter your details below. Your inputs will be used to predict your obesity level.  
""")

def user_input_features():
    st.sidebar.markdown("### Personal Information")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"], help="Select your gender.")
    age = st.sidebar.slider("Age", 10, 80, 30, help="Select your age.")
    height = st.sidebar.slider("Height (in cm)", 130, 200, 170, help="Select your height in cm.")
    weight = st.sidebar.slider("Weight (in kg)", 30, 150, 70, help="Select your weight in kg.")

    st.sidebar.markdown("### Lifestyle Choices")
    family_history = st.sidebar.selectbox("Family History of Obesity", ["Yes", "No"], help="Do you have a family history of obesity?")
    favc = st.sidebar.selectbox("Frequent Consumption of High Caloric Food (FAVC)", ["Yes", "No"], help="Do you frequently consume high-calorie foods?")
    smoke = st.sidebar.selectbox("Smokes?", ["Yes", "No"], help="Do you smoke?")
    scc = st.sidebar.selectbox("Monitor Calories (SCC)?", ["Yes", "No"], help="Do you monitor your calorie intake?")
    faf = st.sidebar.selectbox("Physical Activity (FAF)", ["Low", "Medium", "High"], help="What is your level of physical activity?")
    mtrans = st.sidebar.selectbox("Mode of Transportation (MTRANS)", ["Walking", "Public_Transportation", "Automobile", "Bike", "Motorbike"], help="How do you typically commute?")

    st.sidebar.markdown("### Eating Habits")
    caec = st.sidebar.selectbox("Eating Habit (CAEC)", ["No", "Sometimes", "Frequently", "Always"], help="How often do you eat unhealthy foods?")
    calc = st.sidebar.selectbox("Caloric Intake (CALC)", ["No", "Sometimes", "Frequently", "Always"], help="How often do you consume excess calories?")
    fcvc = st.sidebar.slider("Frequency of Consumption of Vegetables (FCVC)", 1, 3, 2, help="How often do you eat vegetables?")
    ncp = st.sidebar.slider("Number of Meals per Day (NCP)", 1, 5, 3, help="How many meals do you have per day?")
    ch2o = st.sidebar.slider("Daily Water Consumption (CH2O in liters)", 1, 3, 2, help="How much water do you drink daily (in liters)?")
    tue = st.sidebar.slider("Time Using Technology (TUE in hours)", 0, 2, 1, help="How many hours do you spend on technology daily?")

    data = {
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
        "FCVC": fcvc,
        "NCP": ncp,
        "CH2O": ch2o,
        "TUE": tue,
    }
    features = pd.DataFrame(data, index=[0])
    return features

user_input = user_input_features()

# Display user input
st.subheader("User Input Parameters")
st.write(user_input)

# Preprocess the dataset
def preprocess_data(df):
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
    return df

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Obesity prediction.csv") 
    return df

data = load_data()

# Preprocess the dataset
data = preprocess_data(data)

# Separate features and target
X = data.drop(columns=["Obesity"]) 
y = data["Obesity"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree model
clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
clf.fit(X_train, y_train)

# Preprocess user input
preprocessed_input = preprocess_data(user_input)

# Ensure column order matches training data
preprocessed_input = preprocessed_input[X_train.columns]

# Make predictions
try:
    prediction = clf.predict(preprocessed_input)[0]
except Exception as e:
    st.error(f"Prediction Error: {e}")
    st.stop()

# Map prediction to obesity level labels
obesity_levels = {
    "Insufficient_Weight": "Insufficient Weight",
    "Normal_Weight": "Normal Weight",
    "Overweight_Level_I": "Overweight Level I",
    "Overweight_Level_II": "Overweight Level II",
    "Obesity_Type_I": "Obesity Type I",
    "Obesity_Type_II": "Obesity Type II",
    "Obesity_Type_III": "Obesity Type III",
}

prediction_label = obesity_levels.get(prediction, "Unknown")

# Display prediction
st.subheader("Prediction")
st.markdown(f"<h3 style='color: blue;'>Predicted Obesity Level: {prediction_label}</h3>", unsafe_allow_html=True)

# Footer
st.markdown("---")
