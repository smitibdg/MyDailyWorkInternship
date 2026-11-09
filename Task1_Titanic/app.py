"""
Streamlit demo app for Titanic Survival Prediction.
Run:  streamlit run app.py
(First run titanic_survival.py once so titanic_model.pkl exists.)
"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
st.title("🚢 Titanic Survival Predictor")
st.caption("MyDailyWork Data Science Internship - Task 1")

model = joblib.load("titanic_model.pkl")

col1, col2 = st.columns(2)
with col1:
    name_title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])
    sex = st.selectbox("Gender", ["male", "female"])
    age = st.slider("Age", 1, 80, 30)
    pclass = st.selectbox("Ticket class", [1, 2, 3], index=2)
with col2:
    fare = st.number_input("Fare paid", 0.0, 520.0, 15.0)
    sibsp = st.number_input("Siblings / spouses aboard", 0, 8, 0)
    parch = st.number_input("Parents / children aboard", 0, 9, 0)
    embarked = st.selectbox("Port of embarkation", ["S", "C", "Q"])
    has_cabin = st.checkbox("Has a cabin")

if st.button("Predict survival"):
    fam = int(sibsp + parch + 1)
    row = pd.DataFrame([{
        "Age": age, "Fare": fare, "SibSp": sibsp, "Parch": parch,
        "FamilySize": fam, "IsAlone": int(fam == 1), "HasCabin": int(has_cabin),
        "Pclass": str(pclass), "Sex": sex, "Embarked": embarked,
        "Title": name_title, "Deck": "C" if has_cabin else "U",
    }])
    prob = model.predict_proba(row)[0][1]
    st.progress(float(prob))
    st.metric("Survival probability", f"{prob:.1%}")
    if prob >= 0.5:
        st.success("Prediction: SURVIVED ✅")
    else:
        st.error("Prediction: DID NOT SURVIVE ❌")
