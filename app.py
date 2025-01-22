import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import base64

# Set the page configuration (must be first)
st.set_page_config(page_title="Obesity Prediction App", layout="wide")

# Function to encode an image to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Function to set the background
def add_bg_from_local(encoded_image):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{encoded_image}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call function to add background
image_path = "feet.jpg"  # Replace with the path to your background image
encoded_image = encode_image_to_base64(image_path)
add_bg_from_local(encoded_image)

# Add CSS for styling textboxes
st.markdown(
    """
    <style>
    /* Main content font color */
    .stApp {
        color: white;
    }
    /* Sidebar styling: preserves default colors */
    section[data-testid="stSidebar"] {
        color: black !important;
        background-color: #f8f9fa; /* Use the previous background color */
    }
    .textbox {
        background-color: rgba(0, 0, 0, 0.7); /* Semi-transparent black background */
        padding: 10px;
        border-radius: 8px;
        display: inline-block;
    }
    .input-box {
        padding: 15px;
        background-color: rgba(0, 0, 0, 0.5); /* Semi-transparent background for input display */
        border: 1px solid #dcdcdc;
        border-radius: 8px;
        font-family: Arial, sans-serif;
        line-height: 1.6;
        color: white; /* Ensure white font inside input box */
    }
    .input-section-header {
        font-weight: bold;
        margin-bottom: 10px;
        font-size: 16px;
        color: white; /* Section headers inside input box */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Header with textbox
st.markdown(
    """
    <div class="textbox">
        <h1>Obesity Prediction App 🎯</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="textbox">
        <p>
        This app predicts <strong>obesity levels</strong> based on your health and lifestyle inputs.<br>
        Enter your details in the sidebar to get predictions below.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

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
        height = st.slider("Height (in cm)", 130, 198, 170, help="Select your height in cm.")
        weight = st.slider("Weight (in kg)", 30, 173, 70, help="Select your weight in kg.")

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

# Display user input in a styled box
st.markdown(
    f"""
    <div class="textbox">
        <h3>Your Input Parameters</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

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

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Obesity prediction.csv")  # Replace with your dataset
    return df

data = load_data()

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

data = preprocess_data(data)

# Separate features and target
X = data.drop(columns=["Obesity"])  # Replace with your target column name
y = data["Obesity"]  # Replace with your target column name

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
clf.fit(X_train, y_train)

# Preprocess user input
preprocessed_input = preprocess_data(user_input)
preprocessed_input = preprocessed_input[X_train.columns]

# Make prediction
prediction = clf.predict(preprocessed_input)[0]

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

st.markdown("---")

# Display Prediction Header with Textbox
st.markdown(
    """
    <div class="textbox">
        <h3>Prediction</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

# Display Prediction Result
st.markdown(
    f"""
    <div style="padding: 15px; background-color: rgba(0, 0, 0, 0.7); border: 1px solid #d4edda; border-radius: 8px; color: white;">
        🎯 <strong>Predicted Obesity Level:</strong> <span style="color: lightcoral;">{prediction_label}</span>
    </div>
    """,
    unsafe_allow_html=True,
)
