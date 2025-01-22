import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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
st.sidebar.markdown("""Enter your details below. Your inputs will be used to predict your obesity level.""")

# Collapsible Input Sections
def user_input_features():
    # Personal Information Section
    with st.sidebar.expander("Personal Information", expanded=True):
        gender = st.selectbox("Gender", ["Male", "Female"], help="Select your gender.")
        age = st.slider("Age", 10, 80, 30, help="Select your age.")
        height = st.slider("Height (in cm)", 130, 200, 170, help="Select your height in cm.")
        weight = st.slider("Weight (in kg)", 30, 150, 70, help="Select your weight in kg.")

    # Lifestyle Choices Section
    with st.sidebar.expander("Lifestyle Choices"):
        family_history = st.selectbox("Family History of Obesity", ["Yes", "No"], help="Do you have a family history of obesity?")
        favc = st.selectbox("Frequent Consumption of High Caloric Food (FAVC)", ["Yes", "No"], help="Do you frequently consume high-calorie foods?")
        smoke = st.selectbox("Smokes?", ["Yes", "No"], help="Do you smoke?")
        scc = st.selectbox("Monitor Calories (SCC)?", ["Yes", "No"], help="Do you monitor your calorie intake?")
        faf = st.selectbox("Physical Activity (FAF)", ["Low", "Medium", "High"], help="What is your level of physical activity?")
        mtrans = st.selectbox("Mode of Transportation (MTRANS)", ["Walking", "Public_Transportation", "Automobile", "Bike", "Motorbike"], help="How do you typically commute?")

    # Eating Habits Section
    with st.sidebar.expander("Eating Habits"):
        caec = st.selectbox("Eating Habit (CAEC)", ["No", "Sometimes", "Frequently", "Always"], help="How often do you eat unhealthy foods?")
        calc = st.selectbox("Caloric Intake (CALC)", ["No", "Sometimes", "Frequently", "Always"], help="How often do you consume excess calories?")
        fcvc = st.slider("Frequency of Vegetables Consumption (FCVC)", 1, 3, 2, help="How often do you eat vegetables?")
        ncp = st.slider("Number of Meals per Day (NCP)", 1, 5, 3, help="How many meals do you have per day?")
        ch2o = st.slider("Daily Water Consumption (CH2O in liters)", 1, 3, 2, help="How much water do you drink daily (in liters)?")
        tue = st.slider("Time Using Technology (TUE in hours)", 0, 2, 1, help="How many hours do you spend on technology daily?")

    # Combine Inputs into DataFrame
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
    return pd.DataFrame(data, index=[0])

user_input = user_input_features()

# Display user input in a styled box with emojis
st.subheader("Your Input Parameters")

# Add Custom CSS for Styling
st.markdown(
    """
    <style>
    .input-box {
        padding: 15px;
        background-color: #f8f9fa;
        border: 1px solid #dcdcdc;
        border-radius: 8px;
        font-family: Arial, sans-serif;
        line-height: 1.6;
    }
    .input-section-header {
        font-weight: bold;
        margin-bottom: 10px;
        font-size: 16px;
        color: #333333;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display Inputs in a Nicely Formatted Box with Emojis
st.markdown(
    f"""
    <div class="input-box">
        <div class="input-section-header">Personal Information</div>
        <ul>
            <li>👤 <strong>Gender:</strong> {user_input['Gender'][0]}</li>
            <li>🎂 <strong>Age:</strong> {user_input['Age'][0]} years</li>
            <li>📏 <strong>Height:</strong> {user_input['Height'][0] * 100:.1f} cm</li>
            <li>⚖️ <strong>Weight:</strong> {user_input['Weight'][0]} kg</li>
        </ul>
        <div class="input-section-header">Lifestyle Choices</div>
        <ul>
            <li>🏠 <strong>Family History of Obesity:</strong> {user_input['family_history'][0]}</li>
            <li>🍔 <strong>Frequent Consumption of High Caloric Food (FAVC):</strong> {user_input['FAVC'][0]}</li>
            <li>🚬 <strong>Smokes:</strong> {user_input['SMOKE'][0]}</li>
            <li>📊 <strong>Monitor Calories (SCC):</strong> {user_input['SCC'][0]}</li>
            <li>🏃 <strong>Physical Activity Level (FAF):</strong> {user_input['FAF'][0]}</li>
            <li>🚗 <strong>Mode of Transportation (MTRANS):</strong> {user_input['MTRANS'][0]}</li>
        </ul>
        <div class="input-section-header">Eating Habits</div>
        <ul>
            <li>🍩 <strong>Eating Habit (CAEC):</strong> {user_input['CAEC'][0]}</li>
            <li>🍷 <strong>Caloric Intake (CALC):</strong> {user_input['CALC'][0]}</li>
            <li>🥗 <strong>Frequency of Vegetables Consumption (FCVC):</strong> {user_input['FCVC'][0]}</li>
            <li>🍽️ <strong>Number of Meals per Day (NCP):</strong> {user_input['NCP'][0]}</li>
            <li>💧 <strong>Daily Water Consumption (CH2O):</strong> {user_input['CH2O'][0]} liters</li>
            <li>💻 <strong>Time Using Technology (TUE):</strong> {user_input['TUE'][0]} hours</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# Dummy Model Prediction Logic
st.subheader("Prediction")

# Simulated prediction for demonstration
st.markdown(
    f"""
    <div style="padding: 15px; background-color: #e8f5e9; border: 1px solid #d4edda; border-radius: 8px;">
        🎯 <strong>Predicted Obesity Level:</strong> <span style="color: red;">Normal Weight</span>
    </div>
    """,
    unsafe_allow_html=True,
)
