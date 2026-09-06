import streamlit as st
import joblib
import pandas as pd


# Load the trained model and encoders
model = joblib.load("titanic_Logistic_Regression model.pkl")
sex_encoder = joblib.load("sex_encoder.pkl")
embarked_encoder = joblib.load("embarked_encoder.pkl")




st.write("Model expects:")
st.write(model.feature_names_in_)





# Streamlit title
st.title("🚢 Titanic Survival Prediction")

st.write("Enter passenger details to predict survival.")


# User inputs
pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Sex",
    sex_encoder.classes_
)

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=25.0
)

sibsp = st.number_input(
    "Number of Siblings/Spouses",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.number_input(
    "Number of Parents/Children",
    min_value=0,
    max_value=10,
    value=0
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=32.0
)

embarked = st.selectbox(
    "Embarked",
    embarked_encoder.classes_
)


# Prediction button
if st.button("Predict Survival"):

    # Convert categorical values to numbers
    sex_encoded = sex_encoder.transform([sex])[0]
    embarked_encoded = embarked_encoder.transform([embarked])[0]

    # Create input DataFrame
    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex_encoded],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked_encoded]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display result
    if prediction == 1:
        st.success("🎉 The passenger is predicted to SURVIVE.")
    else:
        st.error("❌ The passenger is predicted NOT to survive.")
