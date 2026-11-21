# Iris Flower Classifier - Streamlit app
# Run:  streamlit run app.py   (run the notebook first so iris_model.pkl and iris.csv exist)
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸", layout="wide")

st.markdown("""
<style>
.hero{padding:1.6rem 2rem;border-radius:18px;margin-bottom:1rem;color:white;
      background:linear-gradient(120deg,#3b1c6e 0%,#8e44ad 55%,#e84393 130%);}
.hero h1{margin:0;font-size:2.3rem;color:white}
.hero p{margin:.3rem 0 0 0;opacity:.9}
.card{padding:1rem 1.3rem;border-radius:14px;border:1px solid rgba(128,128,128,.3);background:rgba(142,68,173,.1)}
.big{font-size:2.4rem;font-weight:800;line-height:1.2}
</style>
<div class="hero"><h1>🌸 Iris Flower Classifier</h1>
<p>Move the sliders to describe a flower and the model names its species. MyDailyWork Data Science Internship - Task 3</p></div>
""", unsafe_allow_html=True)

model = joblib.load("iris_model.pkl")
data = pd.read_csv("iris.csv")
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
COLORS = {"setosa": "#e84393", "versicolor": "#8e44ad", "virginica": "#2980b9"}
EMOJI = {"setosa": "🌷", "versicolor": "🪻", "virginica": "🌺"}

# ---------- Sidebar ----------
defaults = {"sepal_length": 5.8, "sepal_width": 3.0, "petal_length": 4.3, "petal_width": 1.3}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("truth", None)


def random_flower():
    r = data.sample(1).iloc[0]
    for f in FEATURES:
        st.session_state[f] = float(r[f])
    st.session_state.truth = r["species"]


def on_slider():
    st.session_state.truth = None


st.sidebar.header("📏 Measurements (cm)")
st.sidebar.button("🎲 Pick a real flower from the dataset", on_click=random_flower)
values = {}
for f in FEATURES:
    lo, hi = float(data[f].min()), float(data[f].max())
    values[f] = st.sidebar.slider(f.replace("_", " ").title(), round(lo - 0.5, 1), round(hi + 0.5, 1),
                                  step=0.1, key=f, on_change=on_slider)

x = pd.DataFrame([values])[FEATURES]
proba = model.predict_proba(x)[0]
classes = list(model.classes_)
species = classes[int(proba.argmax())]

tab1, tab2, tab3 = st.tabs(["🌸 Prediction", "🧊 3D view", "ℹ️ About"])

with tab1:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"""<div class="card"><div class="big">{EMOJI.get(species, '🌸')} Iris {species}</div>
        <p>Confidence: <b>{proba.max() * 100:.0f}%</b></p></div>""", unsafe_allow_html=True)
        if st.session_state.truth:
            ok = st.session_state.truth == species
            if ok:
                st.success(f"Real species: {st.session_state.truth} - model is correct ✅")
            else:
                st.error(f"Real species: {st.session_state.truth} - model got this one wrong ❌")
        bar = go.Figure(go.Bar(x=[p * 100 for p in proba], y=classes, orientation="h",
                               marker_color=[COLORS[c] for c in classes],
                               text=[f"{p * 100:.0f}%" for p in proba], textposition="outside"))
        bar.update_layout(height=220, xaxis=dict(range=[0, 120], title="Probability (%)"), margin=dict(t=10, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(bar)
    with c2:
        fig = px.scatter(data, x="petal_length", y="petal_width", color="species", color_discrete_map=COLORS, opacity=0.6)
        fig.add_trace(go.Scatter(x=[values["petal_length"]], y=[values["petal_width"]], mode="markers",
                                 marker=dict(size=22, symbol="star", color="gold", line=dict(width=2, color="black")),
                                 name="Your flower"))
        fig.update_layout(height=380, margin=dict(t=30, b=10), title="Where your flower sits among the real ones",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig)

with tab2:
    fig = px.scatter_3d(data, x="sepal_length", y="petal_length", z="petal_width", color="species",
                        color_discrete_map=COLORS, opacity=0.7)
    fig.add_trace(go.Scatter3d(x=[values["sepal_length"]], y=[values["petal_length"]], z=[values["petal_width"]],
                               mode="markers", marker=dict(size=9, color="gold", symbol="diamond"), name="Your flower"))
    fig.update_layout(height=560, margin=dict(t=10, b=10))
    st.plotly_chart(fig)
    st.caption("Drag to rotate. The three species form three separate clouds.")

with tab3:
    st.markdown("""
**About this project**
- Data: 150 iris flowers, 4 measurements each, 3 species
- Models compared in the notebook: KNN, Decision Tree, Logistic Regression, Random Forest (5-fold cross validation)
- Fun fact: petal length or petal width alone identifies the species most of the time
- Built with Python, scikit-learn, Plotly and Streamlit
""")
