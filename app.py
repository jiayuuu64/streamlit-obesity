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

# Sidebar with collapsible input sections
st.sidebar.header("User Input Parameters")
st.sidebar.markdown("Enter your details below. Your inputs will be used to predict your obesity level.")

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

# Collating user inputs into a dataframe
def user_input_features():
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

# Display user input in a box
st.subheader("Your Input Parameters")
st.markdown(
    f"""
    <div style="padding: 10px; background-color: #f8f9fa; border: 1px solid #dcdcdc; border-radius: 5px;">
        <strong>Personal Information:</strong><br>
        - Gender: {user_input['Gender'][0]}<br>
        - Age: {user_input['Age'][0]} years<br>
        - Height: {user_input['Height'][0] * 100} cm<br>
        - Weight: {user_input['Weight'][0]} kg<br><br>
        
        <strong>Lifestyle Choices:</strong><br>
        - Family History of Obesity: {user_input['family_history'][0]}<br>
        - Frequent Consumption of High Caloric Food: {user_input['FAVC'][0]}<br>
        - Smokes: {user_input['SMOKE'][0]}<br>
        - Monitor Calories: {user_input['SCC'][0]}<br>
        - Physical Activity Level: {user_input['FAF'][0]}<br>
        - Mode of Transportation: {user_input['MTRANS'][0]}<br><br>
        
        <strong>Eating Habits:</strong><br>
        - Eating Habit: {user_input['CAEC'][0]}<br>
        - Caloric Intake: {user_input['CALC'][0]}<br>
        - Frequency of Vegetables Consumption: {user_input['FCVC'][0]}<br>
        - Number of Meals per Day: {user_input['NCP'][0]}<br>
        - Daily Water Consumption: {user_input['CH2O'][0]} liters<br>
        - Time Using Technology: {user_input['TUE'][0]} hours
    </div>
    """,
    unsafe_allow_html=True,
)


# Display user input in a nicer, side-by-side format
st.subheader("Your Input Parameters")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Personal Information")
    st.markdown(f"**Gender**: {user_input['Gender'][0]}")
    st.markdown(f"**Age**: {user_input['Age'][0]} years")
    st.markdown(f"**Height**: {user_input['Height'][0] * 100} cm")  # Convert height back to cm
    st.markdown(f"**Weight**: {user_input['Weight'][0]} kg")

with col2:
    st.markdown("### Lifestyle Choices")
    st.markdown(f"**Family History of Obesity**: {user_input['family_history'][0]}")
    st.markdown(f"**Frequent Consumption of High Caloric Food (FAVC)**: {user_input['FAVC'][0]}")
    st.markdown(f"**Smokes**: {user_input['SMOKE'][0]}")
    st.markdown(f"**Monitor Calories (SCC)**: {user_input['SCC'][0]}")
    st.markdown(f"**Physical Activity Level (FAF)**: {user_input['FAF'][0]}")
    st.markdown(f"**Mode of Transportation (MTRANS)**: {user_input['MTRANS'][0]}")

st.markdown("### Eating Habits")
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"**Eating Habit (CAEC)**: {user_input['CAEC'][0]}")
    st.markdown(f"**Caloric Intake (CALC)**: {user_input['CALC'][0]}")
    st.markdown(f"**Frequency of Vegetables Consumption (FCVC)**: {user_input['FCVC'][0]}")

with col2:
    st.markdown(f"**Number of Meals per Day (NCP)**: {user_input['NCP'][0]}")
    st.markdown(f"**Daily Water Consumption (CH2O)**: {user_input['CH2O'][0]} liters")
    st.markdown(f"**Time Using Technology (TUE)**: {user_input['TUE'][0]} hours")

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
st.markdown(f"<h3 style='color: blue;'>Predicted Obesity Level: <span style='color: red;'>{prediction_label}</span></h3>", unsafe_allow_html=True)
