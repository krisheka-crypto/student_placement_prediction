import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model/best_model.pkl")
columns = joblib.load("model/model_columns.pkl")

st.title("🎓 Student Placement Predictor")

# Inputs
cgpa = st.number_input("CGPA", 0.0, 10.0, 7.0)
coding = st.number_input("Coding Skill Score", 0, 100, 70)
aptitude = st.number_input("Aptitude Score", 0, 100, 70)
logical = st.number_input("Logical Reasoning Score", 0, 100, 70)
projects = st.number_input("Projects Count", 0, 20, 2)
internships = st.number_input("Internships Count", 0, 10, 1)
backlogs = st.number_input("Backlogs", 0, 10, 0)

# 👇 BUTTON + LOGIC TOGETHER
if st.button("Predict Placement"):

    input_data = pd.DataFrame(columns=columns)
    input_data.loc[0] = 0

    input_data["cgpa"] = cgpa
    input_data["coding_skill_score"] = coding
    input_data["aptitude_score"] = aptitude
    input_data["logical_reasoning_score"] = logical
    input_data["projects_count"] = projects
    input_data["internships_count"] = internships
    input_data["backlogs"] = backlogs

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"🎉 Placed (Confidence: {prob:.2f})")
    else:
        st.error(f"❌ Not Placed (Confidence: {prob:.2f})")