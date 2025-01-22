import streamlit as st
import pandas as pd
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import numpy as np

# App Header
st.set_page_config(page_title="Obesity Prediction App", layout="wide")
st.title("Obesity Prediction App 🎯")
st.markdown("""
This app predicts **obesity levels** based on your health and lifestyle inputs.  
Use the sidebar to enter your details, and view the prediction results below.
""")
st.markdown("---")

# Sidebar
st.sidebar.header("User Input Parameters")
st.sidebar.markdown("""
Enter your details in the fields below and click the button to get predictions.
""")

def user_input_features():
    """Get user inputs from the sidebar."""
    st.sidebar.markdown("### Personal Information")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.slider("Age", 10, 80, 30)
    height = st.sidebar.slider("Height (in cm)", 130, 200, 170)
    weight = st.sidebar.slider("Weight (in kg)", 30, 150, 70)

    st.sidebar.markdown("### Lifestyle Choices")
    family_history = st.sidebar.selectbox("Family History of Obesity", ["Yes", "No"])
    favc = st.sidebar.selectbox("Frequent Consumption of High Caloric Food (FAVC)", ["Yes", "No"])
    smoke = st.sidebar.selectbox("Smokes?", ["Yes", "No"])
    scc = st.sidebar.selectbox("Monitor Calories (SCC)?", ["Yes", "No"])
    faf = st.sidebar.selectbox("Physical Activity (FAF)", ["Low", "Medium", "High"])
    mtrans = st.sidebar.selectbox("Mode of Transportation (MTRANS)", ["Walking", "Public_Transportation", "Automobile", "Bike", "Motorbike"])

    st.sidebar.markdown("### Eating Habits")
    caec = st.sidebar.selectbox("Eating Habit (CAEC)", ["No", "Sometimes", "Frequently", "Always"])
    calc = st.sidebar.selectbox("Caloric Intake (CALC)", ["No", "Sometimes", "Frequently", "Always"])
    fcvc = st.sidebar.slider("Frequency of Consumption of Vegetables (FCVC)", 1, 3, 2)
    ncp = st.sidebar.slider("Number of Meals per Day (NCP)", 1, 5, 3)
    ch2o = st.sidebar.slider("Daily Water Consumption (CH2O in liters)", 1, 3, 2)
    tue = st.sidebar.slider("Time Using Technology (TUE in hours)", 0, 2, 1)

    # Return a DataFrame with the input features
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

# Display user input
st.subheader("User Input Parameters")
st.write(user_input)

# Preprocess the dataset
def preprocess_data(df):
    """Preprocess data by encoding categorical columns."""
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
    """Load obesity prediction dataset."""
    return pd.read_csv("Obesity prediction.csv")

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

# Show confusion matrix (performance of the model on test data)
st.subheader("Model Performance (Confusion Matrix)")

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Convert confusion matrix to a dataframe for Altair charting
cm_df = pd.DataFrame(cm, index=obesity_levels.values(), columns=obesity_levels.values())

# Melt the confusion matrix for Altair plotting
cm_melted = cm_df.reset_index().melt(id_vars="index", var_name="Predicted", value_name="Count")
cm_melted.rename(columns={"index": "True"}, inplace=True)

# Plot confusion matrix using Altair
chart = alt.Chart(cm_melted).mark_rect().encode(
    x=alt.X('Predicted:N', title='Predicted Obesity Level'),
    y=alt.Y('True:N', title='True Obesity Level'),
    color=alt.Color('Count:Q', scale=alt.Scale(scheme='blues'), title='Count'),
    tooltip=['True', 'Predicted', 'Count']
).properties(
    width=600,
    height=400
)

st.altair_chart(chart, use_container_width=True)

# Footer
st.markdown("---")
