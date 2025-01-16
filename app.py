import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Caching the training process
@st.cache_data(allow_output_mutation=True)
def train_model():
    # Load the dataset
    df = pd.read_csv("Obesity prediction.csv")

    # Check the unique values in the Obesity column to adjust the mapping
    st.write("Unique values in 'Obesity' column:", df['Obesity'].unique())

    # Preprocessing the data
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df['family_history'] = le.fit_transform(df['family_history'])
    df['FAVC'] = le.fit_transform(df['FAVC'])
    df['SMOKE'] = le.fit_transform(df['SMOKE'])
    df['SCC'] = le.fit_transform(df['SCC'])
    df = pd.get_dummies(df, columns=['FAF', 'MTRANS', 'CAEC', 'CALC'], drop_first=True)

    # Splitting the dataset into features and target variable
    X = df.drop(columns=['Obesity'])
    y = df['Obesity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initializing and training the model
    dt_model = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=2, random_state=42)
    dt_model.fit(X_train, y_train)
    
    # Return the trained model and the feature columns for reindexing input data
    return dt_model, X.columns

# Train the model (cached)
model, feature_columns = train_model()

# Obesity level map (assuming numeric labels)
# Adjust this map based on the unique values in the 'Obesity' column
obesity_level_map = {
    0: 'Insufficient Weight',
    1: 'Normal Weight',
    2: 'Overweight Level I',
    3: 'Overweight Level II',
    4: 'Obesity Type I',
    5: 'Obesity Type II',
    6: 'Obesity Type III'
}

st.write("""
# Obesity Prediction App
This app predicts the **Obesity Level** based on your inputs!
""")

st.sidebar.header('User Input Parameters')

# User input features
def user_input_features():
    age = st.sidebar.slider('Age', 10, 80, 30)
    height = st.sidebar.slider('Height (in cm)', 130, 200, 160)
    weight = st.sidebar.slider('Weight (in kg)', 30, 150, 70)
    family_history = st.sidebar.selectbox('Family History of Obesity', ['Yes', 'No'])
    FAVC = st.sidebar.selectbox('Frequent Consumption of High Caloric Food (FAVC)', ['Yes', 'No'])
    SMOKE = st.sidebar.selectbox('Smokes?', ['Yes', 'No'])
    SCC = st.sidebar.selectbox('Chronic Disease?', ['Yes', 'No'])
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

# Get user input
df_input = user_input_features()

st.subheader('User Input Parameters')
st.write(df_input)

# Preprocess the input data
le = LabelEncoder()
df_input['family_history'] = le.fit_transform(df_input['family_history'])
df_input['FAVC'] = le.fit_transform(df_input['FAVC'])
df_input['SMOKE'] = le.fit_transform(df_input['SMOKE'])
df_input['SCC'] = le.fit_transform(df_input['SCC'])
df_input['FAF'] = le.fit_transform(df_input['FAF'])
df_input['MTRANS'] = le.fit_transform(df_input['MTRANS'])
df_input['CAEC'] = le.fit_transform(df_input['CAEC'])
df_input['CALC'] = le.fit_transform(df_input['CALC'])

# Ensure the input dataframe has the same columns as the training data
df_input = df_input.reindex(columns=feature_columns, fill_value=0)

# Make a prediction
prediction = model.predict(df_input)

# Get the prediction probabilities
prediction_proba = model.predict_proba(df_input)

# Debugging: Check the shape and contents of prediction and prediction_proba
st.write("Prediction output:", prediction)
st.write("Shape of prediction_proba:", prediction_proba.shape)
st.write("Contents of prediction_proba:", prediction_proba)

# Map prediction to obesity level
predicted_level = obesity_level_map.get(prediction[0], "Unknown")

# Display results
st.subheader('Prediction')
st.write(f'Predicted Obesity Level: {predicted_level}')

# Display prediction probability
st.subheader('Prediction Probability')

# Access the probability for the predicted class
predicted_class_proba = prediction_proba[0][prediction[0]]

st.write(f"Probability of the predicted obesity level: {predicted_class_proba * 100:.2f}%")

st.subheader('Class labels and their corresponding index number')
st.write(obesity_level_map)
