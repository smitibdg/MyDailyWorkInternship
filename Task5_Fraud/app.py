# Credit Card Fraud Detector - Streamlit app
# Run:  streamlit run app.py   (run the notebook first: it creates fraud_model.pkl and demo_transactions.csv)
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Fraud Detector", page_icon="💳", layout="wide")

st.markdown("""
<style>
.hero{padding:1.6rem 2rem;border-radius:18px;margin-bottom:1rem;color:white;
      background:linear-gradient(120deg,#1a1a2e 0%,#16213e 50%,#c0392b 150%);}
.hero h1{margin:0;font-size:2.3rem;color:white}
.hero p{margin:.3rem 0 0 0;opacity:.9}
.card{padding:1rem 1.3rem;border-radius:14px;border:1px solid rgba(128,128,128,.3);background:rgba(192,57,43,.08)}
</style>
<div class="hero"><h1>💳 Credit Card Fraud Radar</h1>
<p>Scan transactions, move the alert threshold and watch the trade-off between missed fraud and false alarms. MyDailyWork Data Science Internship - Task 5</p></div>
""", unsafe_allow_html=True)


@st.cache_resource
def load():
    return joblib.load("fraud_model.pkl")


b = load()
model, scaler, feats = b["model"], b["scaler"], b["features"]

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Controls")
source = st.sidebar.radio("Transactions", ["Demo file (300 transactions)", "Upload my own CSV"])
threshold = st.sidebar.slider("Alert threshold", 0.05, 0.95, 0.50, 0.05,
                              help="A transaction is flagged when its fraud probability is above this value.")
st.sidebar.caption("Low threshold = catch more fraud but more false alarms. High threshold = fewer alarms but more missed fraud.")

if source.startswith("Demo"):
    data = pd.read_csv("demo_transactions.csv")
else:
    up = st.sidebar.file_uploader("CSV with Time, V1..V28, Amount (Class optional)", type="csv")
    data = pd.read_csv(up) if up else None

tab1, tab2, tab3 = st.tabs(["🔎 Scan", "🎭 Why accuracy lies", "ℹ️ About"])

with tab1:
    if data is None:
        st.info("Upload a CSV in the sidebar to start.")
    elif any(c not in data.columns for c in feats):
        st.error("The file is missing some required columns (Time, V1 to V28, Amount).")
    else:
        X = data[feats].copy()
        X[["Time", "Amount"]] = scaler.transform(X[["Time", "Amount"]])   # same scaling as the notebook
        prob = model.predict_proba(X)[:, 1]
        flagged = prob >= threshold

        out = data.copy()
        out["Fraud probability"] = prob.round(3)
        out["Flagged"] = flagged

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Transactions scanned", len(out))
        k2.metric("🚨 Flagged as fraud", int(flagged.sum()))
        k3.metric("Share flagged", f"{flagged.mean() * 100:.1f}%")
        k4.metric("Amount under review", f"${out.loc[flagged, 'Amount'].sum():,.0f}")

        has_truth = "Class" in out.columns
        if has_truth:
            y = out["Class"].values
            tp = int(((flagged) & (y == 1)).sum()); fp = int(((flagged) & (y == 0)).sum())
            fn = int(((~flagged) & (y == 1)).sum()); tn = int(((~flagged) & (y == 0)).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0

            st.subheader("🎯 How good is this threshold? (labels are known in this file)")
            m1, m2, m3 = st.columns(3)
            m1.metric("Frauds caught (recall)", f"{recall * 100:.0f}%", f"{tp} of {tp + fn}")
            m2.metric("Alerts that were real (precision)", f"{precision * 100:.0f}%", f"{tp} of {tp + fp}")
            m3.metric("Genuine wrongly flagged", fp)

            cm_col, hist_col = st.columns(2)
            with cm_col:
                cm = go.Figure(go.Heatmap(z=[[tn, fp], [fn, tp]], x=["Predicted genuine", "Predicted fraud"],
                                          y=["Actually genuine", "Actually fraud"], text=[[tn, fp], [fn, tp]],
                                          texttemplate="%{text}", colorscale="Greens", showscale=False))
                cm.update_layout(height=320, title="Confusion matrix", yaxis_autorange="reversed", margin=dict(t=50, b=10),
                                 paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(cm)
            with hist_col:
                h = go.Figure()
                h.add_histogram(x=prob[y == 0], name="Genuine", marker_color="#1a9100", nbinsx=25, opacity=0.75)
                h.add_histogram(x=prob[y == 1], name="Fraud", marker_color="#c0392b", nbinsx=25, opacity=0.75)
                h.add_vline(x=threshold, line_dash="dash", annotation_text="threshold")
                h.update_layout(barmode="overlay", height=320, title="Fraud probability of each transaction",
                                yaxis_type="log", margin=dict(t=50, b=10), paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(h)

        st.subheader("🚩 Most suspicious transactions")
        show_cols = ["Time", "Amount", "Fraud probability"] + (["Class"] if has_truth else []) + ["Flagged"]
        top = out.sort_values("Fraud probability", ascending=False)[show_cols].head(15)
        st.dataframe(top.style.background_gradient(subset=["Fraud probability"], cmap="Reds"),
                     hide_index=True)
        st.download_button("⬇️ Download all results", out.to_csv(index=False), "fraud_results.csv")

with tab2:
    st.subheader("A lazy model can be 99% accurate and catch zero fraud")
    if data is not None and "Class" in data.columns:
        frauds = int(data["Class"].sum())
        lazy_acc = (1 - data["Class"].mean()) * 100
        st.write(f"In the file you loaded, **{frauds}** of **{len(data)}** transactions are fraud.")
    else:
        lazy_acc, frauds = 99.8, 0
        st.write("In the real dataset only about 0.17% of transactions are fraud.")
    fig = go.Figure(go.Bar(x=["Lazy model: 'everything is genuine'"], y=[lazy_acc], name="Accuracy", marker_color="#7f8c8d",
                           text=[f"{lazy_acc:.1f}% accurate"], textposition="outside"))
    fig.add_bar(x=["Lazy model: 'everything is genuine'"], y=[0], name="Frauds caught", marker_color="#c0392b",
                text=["0% of fraud caught"], textposition="outside")
    fig.update_layout(height=360, yaxis=dict(range=[0, 115], title="%"), margin=dict(t=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig)
    st.markdown("That is why we judge fraud models with **precision** and **recall**, not accuracy.")

with tab3:
    st.markdown(f"""
**About this project**
- Model used: **{b['model_name']}** (chosen by F1-score in the notebook)
- Imbalance handled with class weights, undersampling and oversampling, only on training data
- The anonymised features V1 to V28 come from the original credit card dataset
- Honest note: results here come from a small demo sample, so real-world numbers will differ
""")
