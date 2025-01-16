import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("Obesity prediction.csv")

# Preprocess dataset
le = LabelEncoder()
binary_columns = ['Gender', 'family_history', 'FAVC', 'SMOKE', 'SCC']
for col in binary_columns:
    df[col] = le.fit_transform(df[col])

df = pd.get_dummies(df, columns=['FAF', 'MTRANS', 'CAEC', 'CALC'], drop_first=True)

X = df.drop(columns=['Obesity'])
y = df['Obesity']

# Check unique values in target variable
st.write("Unique values in target (y):", y.unique())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# Save columns for alignment
model_training_columns = X_train.columns

# Map target variable to readable labels
obesity_levels = {
    0: "Insufficient_Weight",
    1: "Normal_Weight",
    2: "Overweight_Level_I",
    3: "Overweight_Level_II",
    4: "Obesity_Type_I",
    5: "Obesity_Type_II",
    6: "Obesity_Type_III",
}

# Sidebar for user input
st.sidebar.header("User Input Parameters")

def user_input_features():
    Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    Age = st.sidebar.slider("Age", 10, 80, 22)
    Height = st.sidebar.slider("Height (in cm)", 130, 200, 178)
    Weight = st.sidebar.slider("Weight (in kg)", 30, 150, 90)
    family_history = st.sidebar.selectbox("Family History of Obesity", ["Yes", "No"])
    FAVC = st.sidebar.selectbox("Frequent Consumption of High Caloric Food (FAVC)", ["Yes", "No"])
    SMOKE = st.sidebar.selectbox("Smokes?", ["Yes", "No"])
    SCC = st.sidebar.selectbox("Chronic Disease?", ["Yes", "No"])
    FAF = st.sidebar.selectbox("Physical Activity (FAF)", ["Low", "Medium", "High", "Frequent"])
    MTRANS = st.sidebar.selectbox("Mode of Transportation (MTRANS)", ["Public_Transportation", "Bike", "Motorbike", "Walking", "Automobile"])
    CAEC = st.sidebar.selectbox("Eating Habit (CAEC)", ["No", "Sometimes", "Frequently", "Always"])
    CALC = st.sidebar.selectbox("Caloric Intake (CALC)", ["No", "Sometimes", "Frequently", "Always"])

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
        "CALC": CALC,
    }
    return data

# Preprocess user input
def preprocess_user_input(user_input):
    df_input = pd.DataFrame([user_input])

    # Debugging: Check which columns exist
    st.write("Columns in df_input before encoding:", df_input.columns)

    # Encode binary columns
    for col in binary_columns:
        if col in df_input.columns:  # Check if the column exists
            df_input[col] = le.fit_transform(df_input[col])
        else:
            st.error(f"Column '{col}' is missing in user input!")
            raise KeyError(f"Column '{col}' is missing in user input!")

    # Apply one-hot encoding
    df_input = pd.get_dummies(df_input, columns=['FAF', 'MTRANS', 'CAEC', 'CALC'], drop_first=True)

    # Align columns with training data
    missing_cols = set(model_training_columns) - set(df_input.columns)
    for col in missing_cols:
        df_input[col] = 0
    df_input = df_input[model_training_columns]

    return df_input

user_input = user_input_features()
st.subheader("User Input Parameters")
st.write(pd.DataFrame([user_input]))

# Preprocess user input and predict
preprocessed_input = preprocess_user_input(user_input)
prediction = model.predict(preprocessed_input)
prediction_proba = model.predict_proba(preprocessed_input)

# Debugging: Raw prediction output
st.write(f"Raw prediction output: {prediction[0]}")

# Make prediction and display results
st.subheader("Prediction")
if prediction[0] in obesity_levels:
    st.write(f"Predicted Obesity Level: {obesity_levels[prediction[0]]}")
else:
    st.error(f"Unexpected prediction value: {prediction[0]}")
    st.write("Ensure the model output matches the obesity_levels keys.")

st.subheader("Prediction Probability")
st.write(f"Probability of the predicted obesity level: {prediction_proba[0][prediction[0]] * 100:.2f}%")

st.subheader("Class labels and their corresponding index number")
st.write(obesity_levels)
