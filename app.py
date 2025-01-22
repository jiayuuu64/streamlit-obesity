import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# App Header
st.set_page_config(page_title="Obesity Prediction App", layout="wide")
st.title("Obesity Prediction App 🎯")
st.markdown("""
This app predicts **obesity levels** based on your health and lifestyle inputs.  
Use the sidebar to enter your details, and view the prediction results below.  
""")

st.markdown("---")

# Adding background image using CSS
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("/image/weight.jpg");
        background-size: cover;
        background-position: center;
    }
    .rectangle-box {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.header("User Input Parameters")
st.sidebar.markdown("""
Enter your details in the fields below and click the button to get predictions.
""")

def user_input_features():
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

# Display user input in a rectangle box with a background color
st.markdown('<div class="rectangle-box">', unsafe_allow_html=True)
st.subheader("User Input Parameters")

# Styling the display to be side-by-side
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"**Gender**: {user_input['Gender'][0]}")
    st.markdown(f"**Age**: {user_input['Age'][0]} years")
    st.markdown(f"**Height**: {user_input['Height'][0]} m")
    st.markdown(f"**Weight**: {user_input['Weight'][0]} kg")

with col2:
    st.markdown(f"**Family History of Obesity**: {user_input['family_history'][0]}")
    st.markdown(f"**Frequent Caloric Food**: {user_input['FAVC'][0]}")
    st.markdown(f"**Smokes**: {user_input['SMOKE'][0]}")
    st.markdown(f"**Monitor Calories**: {user_input['SCC'][0]}")
    st.markdown(f"**Physical Activity**: {user_input['FAF'][0]}")
    st.markdown(f"**Transportation Mode**: {user_input['MTRANS'][0]}")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Load dataset
def load_data():
    df = pd.read_csv("Obesity prediction.csv")
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

# Display prediction in a styled rectangle box
st.markdown('<div class="rectangle-box">', unsafe_allow_html=True)
st.subheader("Prediction")
st.markdown(f"<h3 style='color: blue;'>Predicted Obesity Level: {prediction_label}</h3>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
