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

# Example Prediction Section (Model loading and predictions remain unchanged)
st.subheader("Prediction")
st.write("Predicted Obesity Level: Overweight Level II")  # Example placeholder prediction

# Example Visualization
st.subheader("Prediction Probability")
st.markdown("### Here's how likely you are to belong to each obesity category:")
fig, ax = plt.subplots()
ax.bar(["Insufficient", "Normal", "Overweight I", "Overweight II", "Obesity I", "Obesity II", "Obesity III"], [5, 10, 15, 70, 5, 3, 2])
st.pyplot(fig)
st.success("Prediction complete!")
st.balloons()
