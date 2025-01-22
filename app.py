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
            text-align: center; /* Align everything to center */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call function to add background
image_path = "feet.jpg"  # Replace with the path to your background image
encoded_image = encode_image_to_base64(image_path)
add_bg_from_local(encoded_image)

# Add CSS for styling
st.markdown(
    """
    <style>
    .textbox {
        background-color: rgba(0, 0, 0, 0.75);
        padding: 15px 20px;
        border-radius: 15px;
        margin-bottom: 10px;
        width: fit-content;
        margin-left: auto;
        margin-right: auto;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        color: white;
        text-align: center;
    }
    .input-box {
        padding: 20px;
        background-color: rgba(0, 0, 0, 0.7);
        border-radius: 15px;
        border: none;
        margin-bottom: 10px;
        margin-left: auto;
        margin-right: auto;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        font-family: Arial, sans-serif;
        line-height: 1.6;
        color: white;
        text-align: left;
    }
    .input-section-header {
        font-weight: bold;
        margin-bottom: 10px;
        font-size: 18px;
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Header with improved textbox
st.markdown(
    """
    <div class="textbox">
        <h1>Obesity Prediction App 🎯</h1>
        <p>This app predicts <strong>obesity levels</strong> based on your health and lifestyle inputs. Enter your details in the sidebar to get predictions below.</p>
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
        age = st.slider("Age", 10, 80, 30, help="Enter your age.")
        height = st.slider("Height (in meters)", 1.45, 1.98, 1.70, help="Enter your height in meters.")
        weight = st.slider("Weight (in kg)", 39.0, 173.0, 70.0, help="Enter your weight in kilograms.")

    # Lifestyle Choices Section
    with st.sidebar.expander("Lifestyle Choices"):
        family_history = st.selectbox("Family History of Obesity", ["Yes", "No"], help="Has a family member suffered or suffers from overweight?")
        favc = st.selectbox("Do you eat high caloric food frequently?", ["Yes", "No"], help="Do you consume high-calorie foods often?")
        smoke = st.selectbox("Do you smoke?", ["Yes", "No"], help="Do you smoke regularly?")
        scc = st.selectbox("Do you monitor the calories you eat daily?", ["Yes", "No"], help="Do you track your calorie intake daily?")
        faf = st.selectbox("How often do you have physical activity?", ["Low", "Medium", "High"], help="Rate your physical activity level.")
        mtrans = st.selectbox("Which transportation do you usually use?", ["Walking", "Public_Transportation", "Automobile", "Bike", "Motorbike"], help="Select your most used mode of transportation.")

    # Eating Habits Section
    with st.sidebar.expander("Eating Habits"):
        caec = st.selectbox("Do you eat any food between meals?", ["No", "Sometimes", "Frequently", "Always"], help="How often do you eat between meals?")
        calc = st.selectbox("How often do you drink alcohol?", ["No", "Sometimes", "Frequently", "Always"], help="How often do you consume alcohol?")
        fcvc = st.slider("How often do you eat vegetables?", 1, 3, 2, help="Frequency of vegetables in your diet.")
        ncp = st.slider("How many main meals do you have daily?", 1, 5, 3, help="Number of main meals you consume daily.")
        ch2o = st.slider("How much water do you drink daily (in liters)?", 1, 3, 2, help="Liters of water consumed daily.")
        tue = st.slider("How much time do you use technology daily (in hours)?", 0, 2, 1, help="Hours spent on technological devices daily.")

    # Combine Inputs into DataFrame
    data = {
        "Gender": gender,
        "Age": age,
        "Height": height,
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
X = data.drop(columns=["Obesity_level"])  # Replace with your target column name
y = data["Obesity_level"]  # Replace with your target column name

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

# Display Prediction
st.markdown(
    f"""
    <div class="textbox">
        🎯 <strong>Predicted Obesity Level:</strong> <span style="color:red; font-size: 24px;">{prediction}</span>
    </div>
    """,
    unsafe_allow_html=True,
)
