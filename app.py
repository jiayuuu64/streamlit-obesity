import streamlit as st
import pandas as pd
import joblib

# Title for the app
st.write("""
# Obesity Level Prediction App
This app predicts the **Obesity Level** based on user input features!
""")

# Sidebar for user input parameters
st.sidebar.header('User Input Parameters')

def user_input_features():
    # Get user inputs for each feature
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    age = st.sidebar.slider('Age', 10, 100, 25)
    height = st.sidebar.slider('Height (cm)', 100, 250, 170)
    weight = st.sidebar.slider('Weight (kg)', 30, 200, 70)
    family_history = st.sidebar.selectbox('Family History of Obesity', ['Yes', 'No'])
    FAVC = st.sidebar.selectbox('Frequent consumption of high calorie food', ['Yes', 'No'])
    FCVC = st.sidebar.slider('Frequency of vegetable consumption', 1, 10, 5)
    NCP = st.sidebar.slider('Number of main meals per day', 1, 5, 3)
    CAEC = st.sidebar.slider('Consumption of food between meals', 1, 10, 5)
    SMOKE = st.sidebar.selectbox('Smoking', ['Yes', 'No'])
    CH2O = st.sidebar.slider('Water consumption (l/day)', 1, 10, 2)
    SCC = st.sidebar.slider('Calories consumption', 1000, 5000, 2000)
    FAF = st.sidebar.slider('Physical activity frequency (hours/week)', 0, 10, 3)
    TUE = st.sidebar.slider('Time spent in physical activity (minutes/day)', 0, 300, 60)
    MTRANS = st.sidebar.selectbox('Transportation', ['Walking', 'Bike', 'Public transport', 'Automobile'])

    # Map categorical variables to numeric values
    gender = 1 if gender == 'Male' else 0
    family_history = 1 if family_history == 'Yes' else 0
    FAVC = 1 if FAVC == 'Yes' else 0
    SMOKE = 1 if SMOKE == 'Yes' else 0
    MTRANS = {'Walking': 0, 'Bike': 1, 'Public transport': 2, 'Automobile': 3}[MTRANS]

    # Create a DataFrame with the user input
    data = {
        'Gender': gender,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'family_history': family_history,
        'FAVC': FAVC,
        'FCVC': FCVC,
        'NCP': NCP,
        'CAEC': CAEC,
        'SMOKE': SMOKE,
        'CH2O': CH2O,
        'SCC': SCC,
        'FAF': FAF,
        'TUE': TUE,
        'MTRANS': MTRANS
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get the user input
df = user_input_features()

# Show the user input parameters
st.subheader('User Input parameters')
st.write(df)

# Load the pre-trained model (assuming 'obesity_model.pkl' is saved in the current directory)
model = joblib.load('obesity_model.pkl')

# Make prediction using the loaded model
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

# Show obesity level prediction
st.subheader('Prediction')
st.write(f'Predicted Obesity Level: {prediction[0]}')

# Show prediction probabilities
st.subheader('Prediction Probability')
st.write(prediction_proba)

# Display the classification labels
st.subheader('Obesity Levels')
st.write(['Insufficient Weight', 'Normal Weight', 'Overweight Level I', 'Overweight Level II', 
          'Obesity Type I', 'Obesity Type II', 'Obesity Type III'])
