import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Cache the training process
@st.cache_data
def train_model():
    # Load the dataset
    df = pd.read_csv("Obesity prediction.csv")

    # Preprocessing
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df['family_history'] = le.fit_transform(df['family_history'])
    df['FAVC'] = le.fit_transform(df['FAVC'])
    df['SMOKE'] = le.fit_transform(df['SMOKE'])
    df['SCC'] = le.fit_transform(df['SCC'])
    df = pd.get_dummies(df, columns=['FAF', 'MTRANS', 'CAEC', 'CALC'], drop_first=True)

    # Features and target
    X = df.drop(columns=['Obesity'])
    y = df['Obesity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    dt_model = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=2, random_state=42)
    dt_model.fit(X_train, y_train)

    return dt_model, X.columns, dt_model.classes_

# Train the model
model, feature_columns, class_labels = train_model()

# Create the obesity level map dynamically from model's class labels
obesity_level_map = {i: label for i, label in enumerate(class_labels)}

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

# Preprocess the input
le = LabelEncoder()
df_input['family_history'] = le.fit_transform(df_input['family_history'])
df_input['FAVC'] = le.fit_transform(df_input['FAVC'])
df_input['SMOKE'] = le.fit_transform(df_input['SMOKE'])
df_input['SCC'] = le.fit_transform(df_input['SCC'])
df_input['FAF'] = le.fit_transform(df_input['FAF'])
df_input['MTRANS'] = le.fit_transform(df_input['MTRANS'])
df_input['CAEC'] = le.fit_transform(df_input['CAEC'])
df_input['CALC'] = le.fit_transform(df_input['CALC'])

# Ensure the input matches the training features
df_input = df_input.reindex(columns=feature_columns, fill_value=0)

# Debugging Steps
st.write("### Debugging Information:")
st.write("Raw User Input Data:")
st.write(df_input)

st.write("Feature Alignment Check:")
st.write("Feature Columns in Model Training Order:", feature_columns)
st.write("Columns in App Input Data:", df_input.columns)

# Uncomment if comparing with dataset
# st.write("Original Dataset Row (for comparison):")
# st.write(df.iloc[ROW_INDEX])  # Replace ROW_INDEX with the index of the dataset row you want to compare.

# Prediction
prediction = model.predict(df_input)[0]  # Get the predicted class label

# Get the prediction probabilities
prediction_proba = model.predict_proba(df_input)

# Map prediction to obesity level
predicted_level = obesity_level_map.get(np.where(class_labels == prediction)[0][0], "Unknown")

# Display prediction
st.subheader('Prediction')
st.write(f'Predicted Obesity Level: {predicted_level}')

# Display prediction probability
st.subheader('Prediction Probability')
try:
    predicted_class_proba = prediction_proba[0][np.where(class_labels == prediction)[0][0]]
    st.write(f"Probability of the predicted obesity level: {predicted_class_proba * 100:.2f}%")
except IndexError:
    st.write("Error: Unable to access probability for the predicted class.")

# Display class mapping
st.subheader('Class labels and their corresponding index number')
st.write(obesity_level_map)
