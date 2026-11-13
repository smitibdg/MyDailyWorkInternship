# Titanic Survival Predictor - Streamlit app
# Run:  streamlit run app.py   (run the notebook first so titanic_model.pkl exists)
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="wide")

# ---------- Look and feel ----------
st.markdown("""
<style>
.hero{padding:1.6rem 2rem;border-radius:18px;margin-bottom:1rem;color:white;
      background:linear-gradient(120deg,#06233f 0%,#0b5c8a 60%,#1a9100 140%);}
.hero h1{margin:0;font-size:2.3rem;color:white}
.hero p{margin:.3rem 0 0 0;opacity:.9}
.card{padding:1rem 1.3rem;border-radius:14px;border:1px solid rgba(128,128,128,.3);
      background:rgba(11,92,138,.08);margin-bottom:.7rem}
.ok{border-left:8px solid #1a9100}.bad{border-left:8px solid #d62728}
.big{font-size:2rem;font-weight:800;line-height:1.2}
</style>
<div class="hero"><h1>🚢 Would You Have Survived the Titanic?</h1>
<p>Enter a ticket in the sidebar and a Random Forest model predicts your fate. MyDailyWork Data Science Internship - Task 1</p></div>
""", unsafe_allow_html=True)

model = joblib.load("titanic_model.pkl")
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


def survival_chance(pclass, sex, age, sibsp, parch, fare, embarked):
    """Returns the model's survival probability (0 to 1) for one passenger."""
    row = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]], columns=FEATURES)
    return float(model.predict_proba(row)[0][list(model.classes_).index(1)])


# ---------- Sidebar: the passenger ----------
st.sidebar.header("🎟️ Your ticket")
name = st.sidebar.text_input("Passenger name", "Jack")
sex_label = st.sidebar.radio("Gender", ["Male", "Female"], horizontal=True)
age = st.sidebar.slider("Age", 1, 80, 28)
pclass = st.sidebar.radio("Ticket class", [1, 2, 3], horizontal=True,
                          format_func=lambda c: {1: "1st", 2: "2nd", 3: "3rd"}[c])
fare = st.sidebar.slider("Fare paid (£)", 0, 300, {1: 85, 2: 20, 3: 8}[pclass])
sibsp = st.sidebar.number_input("Siblings / spouse aboard", 0, 8, 0)
parch = st.sidebar.number_input("Parents / children aboard", 0, 6, 0)
port = st.sidebar.selectbox("Boarded at", ["Southampton", "Cherbourg", "Queenstown"])

sex = 1 if sex_label == "Female" else 0
embarked = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}[port]
chance = survival_chance(pclass, sex, age, sibsp, parch, fare, embarked)

tab1, tab2, tab3 = st.tabs(["🎟️ Prediction", "🧠 What drove the model", "ℹ️ About"])

# ---------- Tab 1: prediction ----------
with tab1:
    left, right = st.columns([1, 1])
    with left:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=chance * 100, number={"suffix": "%"},
            title={"text": "Survival chance"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#0b5c8a"},
                   "steps": [{"range": [0, 40], "color": "#f4c7c3"},
                             {"range": [40, 60], "color": "#fde9b8"},
                             {"range": [60, 100], "color": "#c6e8c0"}]}))
        fig.update_layout(height=300, margin=dict(t=60, b=10, l=30, r=30), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig)
    with right:
        survived = chance >= 0.5
        st.markdown(
            f"""<div class="card {'ok' if survived else 'bad'}">
            <div class="big">{'✅ ' + name + ' SURVIVES' if survived else '❌ ' + name + ' does not survive'}</div>
            <p>{'A seat on a lifeboat was found.' if survived else 'The lifeboats ran out of room.'}
            Model confidence: <b>{max(chance, 1 - chance) * 100:.0f}%</b></p></div>""",
            unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Class", f"{pclass}{'st' if pclass == 1 else 'nd' if pclass == 2 else 'rd'}")
        c2.metric("Fare", f"£{fare}")
        c3.metric("Family aboard", int(sibsp + parch))

    st.subheader("🔀 What if something was different?")
    scenarios = {
        "Your ticket": chance,
        "Opposite gender": survival_chance(pclass, 1 - sex, age, sibsp, parch, fare, embarked),
        "1st class": survival_chance(1, sex, age, sibsp, parch, max(fare, 85), embarked),
        "3rd class": survival_chance(3, sex, age, sibsp, parch, min(fare, 8), embarked),
        "Travelling alone": survival_chance(pclass, sex, age, 0, 0, fare, embarked),
    }
    bar = go.Figure(go.Bar(x=list(scenarios.keys()), y=[v * 100 for v in scenarios.values()],
                           marker_color=["#0b5c8a"] + ["#8fb8d1"] * 4,
                           text=[f"{v * 100:.0f}%" for v in scenarios.values()], textposition="outside"))
    bar.update_layout(height=340, yaxis=dict(range=[0, 115], title="Survival chance (%)"),
                      margin=dict(t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(bar)
    st.caption("In this dataset gender decides survival almost completely, so the 'Opposite gender' bar flips.")

# ---------- Tab 2: feature importance ----------
with tab2:
    st.subheader("Which details mattered most?")
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color="#1a9100"))
        fig.update_layout(height=380, margin=dict(t=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig)
        st.info(f"Top feature: **{imp.index[-1]}**. Taller bars = the model relied on that detail more.")
    else:
        st.write("This model does not provide feature importance.")

# ---------- Tab 3: about ----------
with tab3:
    st.markdown("""
**About this project**
- Data: 418 Titanic passengers (age, gender, class, fare, family, port) and whether they survived
- Model: Random Forest trained in the notebook, compared with Naive Bayes and Logistic Regression
- Honest note: in this file every woman survived and every man died, so accuracy is about 100%.
  On the bigger original Titanic dataset the same idea scores around 80%.
- Built with Python, scikit-learn, Plotly and Streamlit
""")
