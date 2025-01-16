import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load pre-trained model
model = joblib.load('obesity_model.pkl')

st.write("""
# Obesity Prediction App
This app predicts the **Obesity Level** based on your inputs!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    # Add sliders or input fields for each feature
    age = st.sidebar.slider('Age', 10, 80, 30)
    height = st.sidebar.slider('Height (in cm)', 130, 200, 160)
    weight = st.sidebar.slider('Weight (in kg)', 30, 150, 70)
    family_history = st.sidebar.selectbox('Family History of Obesity', ['Yes', 'No'])
    FAVC = st.sidebar.selectbox('Frequent Consumption of High Caloric Food (FAVC)', ['Yes', 'No'])
    SMOKE = st.sidebar.selectbox('Smokes?', ['Yes', 'No'])
    SCC = st.sidebar.selectbox('Chronic Disease?', ['Yes', 'No'])
    
    # Add other relevant inputs like FAF, MTRANS, CAEC, CALC, etc.
    FAF = st.sidebar.selectbox('Physical Activity (FAF)', ['Low', 'Medium', 'High'])
    MTRANS = st.sidebar.selectbox('Mode of Transportation (MTRANS)', ['Walking', 'Bicycle', 'Public', 'Private'])
    CAEC = st.sidebar.selectbox('Eating Habit (CAEC)', ['Low', 'Medium', 'High'])
    CALC = st.sidebar.selectbox('Caloric Intake (CALC)', ['Low', 'Medium', 'High'])

    data = {
        'Age': age,
        'Height': height,
        'Weight': weight,
        'family_history': family_history,
        'FAVC': FAVC,
        'SMOKE': SMOKE,
        'SCC': SCC,
        'FAF': FAF,
        'MTRANS': MTRANS,
        'CAEC': CAEC,
        'CALC': CALC
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input features
df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

# Preprocessing: Encode categorical features as needed
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['family_history'] = le.fit_transform(df['family_history'])
df['FAVC'] = le.fit_transform(df['FAVC'])
df['SMOKE'] = le.fit_transform(df['SMOKE'])
df['SCC'] = le.fit_transform(df['SCC'])
df['FAF'] = le.fit_transform(df['FAF'])
df['MTRANS'] = le.fit_transform(df['MTRANS'])
df['CAEC'] = le.fit_transform(df['CAEC'])
df['CALC'] = le.fit_transform(df['CALC'])

# Model Prediction
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

# Display the result
st.subheader('Prediction')
st.write(f'Obesity Level: {prediction[0]}')

st.subheader('Prediction Probability')
st.write(prediction_proba)

# Optionally, map the predictions to actual obesity levels
obesity_levels = ['Insufficient Weight', 'Normal Weight', 'Overweight Level I', 'Overweight Level II', 
                  'Obesity Type I', 'Obesity Type II', 'Obesity Type III']

st.subheader('Class labels and their corresponding index number')
st.write(obesity_levels)

# Show detailed prediction result with probability
st.write(f"Probability of the predicted obesity level: {prediction_proba[0][prediction[0]]*100:.2f}%")
