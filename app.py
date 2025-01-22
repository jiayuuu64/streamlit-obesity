import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# App Header
st.title("Obesity Prediction App 🎯")
st.markdown("This app predicts the **obesity level** based on your health and lifestyle choices.")
st.markdown("---")

# Sidebar
st.sidebar.header("User Input Parameters")

def user_input_features():
    with st.sidebar.expander("Personal Information", expanded=True):
        gender = st.selectbox("Gender", ["Male", "Female"], help="Your biological gender.")
        age = st.slider("Age", 10, 80, 30, help="Your age in years.")
        height = st.slider("Height (in cm)", 130, 200, 170, help="Your height in centimeters.")
        weight = st.slider("Weight (in kg)", 30, 150, 70, help="Your weight in kilograms.")

    with st.sidebar.expander("Lifestyle Choices", expanded=True):
        family_history = st.selectbox("Family History of Obesity", ["Yes", "No"], help="Does obesity run in your family?")
        favc = st.selectbox("Frequent Consumption of High Caloric Food (FAVC)", ["Yes", "No"])
        smoke = st.selectbox("Smokes?", ["Yes", "No"])
        scc = st.selectbox("Monitor Calories (SCC)?", ["Yes", "No"])
        faf = st.selectbox("Physical Activity (FAF)", ["Low", "Medium", "High"])
        mtrans = st.selectbox("Mode of Transportation (MTRANS)", ["Walking", "Public_Transportation", "Automobile", "Bike", "Motorbike"])

    with st.sidebar.expander("Eating Habits", expanded=True):
        caec = st.selectbox("Eating Habit (CAEC)", ["No", "Sometimes", "Frequently", "Always"])
        calc = st.selectbox("Caloric Intake (CALC)", ["No", "Sometimes", "Frequently", "Always"])
        fcvc = st.slider("Frequency of Consumption of Vegetables (FCVC)", 1, 3, 2)
        ncp = st.slider("Number of Meals per Day (NCP)", 1, 5, 3)
        ch2o = st.slider("Daily Water Consumption (CH2O in liters)", 1, 3, 2)
        tue = st.slider("Time Using Technology (TUE in hours)", 0, 2, 1)

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

# Display User Inputs
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
    prediction_proba = clf.predict_proba(preprocessed_input)[0]
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
st.write(f"Predicted Obesity Level: {prediction_label}")

# Display prediction probability using Streamlit's bar chart
st.subheader("Prediction Probability")
st.markdown("### Here's how likely you are to belong to each obesity category:")
probability_df = pd.DataFrame({
    "Obesity Level": [obesity_levels.get(level, level) for level in clf.classes_],
    "Probability (%)": prediction_proba * 100,
})
st.bar_chart(probability_df.set_index("Obesity Level"))

# Add Feedback
st.success("Prediction complete! 🎉")
st.balloons()
