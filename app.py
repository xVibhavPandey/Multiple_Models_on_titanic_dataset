import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page configuration
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details below to predict survival probability.")

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("model5.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Error: `model5.pkl` not found in the current directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Input layout
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=2)
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0, step=1.0)
    sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)

with col2:
    parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare Paid ($)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)
    embarked = st.selectbox("Port of Embarkation", ["S (Southampton)", "C (Cherbourg)", "Q (Queenstown)"])

# Format categorical inputs
sex_encoded = 1 if sex == "male" else 0
embarked_char = embarked[0]  # 'S', 'C', or 'Q'
embarked_map = {"S": 0, "C": 1, "Q": 2}
embarked_encoded = embarked_map[embarked_char]

# Prepare input DataFrame matching training features
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex_encoded],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked_encoded]
})

# Prediction trigger
if st.button("Predict Survival", type="primary"):
    try:
        prediction = model.predict(input_data)[0]
        
        # Check if predict_proba is supported
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1] * 100
            st.metric("Survival Probability", f"{probability:.2f}%")

        if prediction == 1:
            st.success("Result: The passenger is predicted to **SURVIVE**.")
        else:
            st.error("Result: The passenger is predicted to **NOT SURVIVE**.")
            
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Check if the column names/order match what was used to fit `model5` in your notebook.")