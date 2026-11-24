# Sales Predictor & Budget Optimizer - Streamlit app
# Run:  streamlit run app.py   (run the notebook first so sales_model.pkl exists)
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Sales Predictor", page_icon="📈", layout="wide")

st.markdown("""
<style>
.hero{padding:1.6rem 2rem;border-radius:18px;margin-bottom:1rem;color:white;
      background:linear-gradient(120deg,#0b3d02 0%,#1a9100 60%,#f5b50f 150%);}
.hero h1{margin:0;font-size:2.3rem;color:white}
.hero p{margin:.3rem 0 0 0;opacity:.9}
.card{padding:1rem 1.3rem;border-radius:14px;border:1px solid rgba(128,128,128,.3);background:rgba(26,145,0,.08)}
.big{font-size:3rem;font-weight:800;line-height:1.1}
</style>
<div class="hero"><h1>📈 Where Should the Ad Money Go?</h1>
<p>Set a budget for TV, Radio and Newspaper, see predicted sales, then let the model find the best split. MyDailyWork Data Science Internship - Task 4</p></div>
""", unsafe_allow_html=True)

b = joblib.load("sales_model.pkl")
model, feats, ranges, avg_sales = b["model"], b["features"], b["ranges"], b["average_sales"]
COLORS = ["#1a9100", "#f5b50f", "#7f8c8d"]


def predict(df):
    return model.predict(df[feats])


# ---------- Sidebar ----------
st.sidebar.header("💰 Ad budget (dataset units)")
budget = {}
for f in feats:
    lo, hi = ranges[f]
    budget[f] = st.sidebar.slider(f"{f} spend", float(lo), float(hi), float(round((lo + hi) / 2, 1)))
current = pd.DataFrame([budget])
pred = float(predict(current)[0])

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📉 Channel response", "🤖 Budget optimizer", "ℹ️ About"])

# ---------- Tab 1 ----------
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="card"><p>Predicted sales</p><div class="big">{pred:.1f}</div></div>""", unsafe_allow_html=True)
    c2.metric("Average sales in data", f"{avg_sales:.1f}", f"{pred - avg_sales:+.1f} vs average")
    c3.metric("Total ad budget", f"{sum(budget.values()):.0f}")

    st.subheader("🔀 What if I add 20% to one channel?")
    deltas = []
    for f in feats:
        more = current.copy()
        more[f] = min(more[f].iloc[0] * 1.2, ranges[f][1] * 1.2)
        deltas.append(float(predict(more)[0]) - pred)
    fig = go.Figure(go.Bar(x=feats, y=deltas, marker_color=COLORS[:len(feats)],
                           text=[f"{d:+.2f}" for d in deltas], textposition="outside"))
    fig.update_layout(height=340, yaxis_title="Change in predicted sales", margin=dict(t=20, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig)
    best = feats[int(np.argmax(deltas))]
    st.success(f"Best channel to boost right now: **{best}**")

# ---------- Tab 2 ----------
with tab2:
    st.subheader("How sales react as one channel's spend changes")
    st.caption("The other two channels stay at your current slider values.")
    fig = go.Figure()
    for f, col in zip(feats, COLORS):
        grid = np.linspace(ranges[f][0], ranges[f][1], 40)
        df = pd.concat([current] * len(grid), ignore_index=True)
        df[f] = grid
        fig.add_trace(go.Scatter(x=grid / ranges[f][1] * 100, y=predict(df), mode="lines", name=f,
                                 line=dict(color=col, width=3)))
    fig.update_layout(height=420, xaxis_title="Spend (% of the channel's maximum)", yaxis_title="Predicted sales",
                      margin=dict(t=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig)
    st.caption("A steep line means that channel pays off. A flat line means extra money is wasted there.")

# ---------- Tab 3 ----------
with tab3:
    st.subheader("Give the model a total budget and it searches the best split")
    total = st.slider("Total budget", float(sum(r[0] for r in ranges.values())),
                      float(sum(r[1] for r in ranges.values())), float(sum(budget.values())))
    if len(feats) == 3:
        g0, g1 = [np.linspace(ranges[f][0], ranges[f][1], 40) for f in feats[:2]]
        a, c = np.meshgrid(g0, g1)
        third = total - a - c
        lo3, hi3 = ranges[feats[2]]
        ok = (third >= lo3) & (third <= hi3)
        if ok.any():
            grid = pd.DataFrame({feats[0]: a[ok], feats[1]: c[ok], feats[2]: third[ok]})
            grid["pred"] = predict(grid)
            top = grid.loc[grid["pred"].idxmax()]
            cur_total = pd.DataFrame([budget])
            scale = total / cur_total.iloc[0].sum()
            yours = cur_total * scale
            yours_pred = float(predict(yours)[0])
            m1, m2, m3 = st.columns(3)
            m1.metric("Your split (scaled to this budget)", f"{yours_pred:.1f}")
            m2.metric("Model's best split", f"{top['pred']:.1f}", f"{top['pred'] - yours_pred:+.1f}")
            m3.metric("Gain", f"{(top['pred'] / yours_pred - 1) * 100:+.1f}%")
            fig = go.Figure()
            fig.add_bar(name="Your split", x=feats, y=yours.iloc[0].values, marker_color="#7f8c8d")
            fig.add_bar(name="Best split", x=feats, y=[top[f] for f in feats], marker_color="#1a9100")
            fig.update_layout(barmode="group", height=360, yaxis_title="Spend", margin=dict(t=20),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig)
        else:
            st.warning("No valid split for this total. Try another budget.")
    else:
        st.info("The optimizer expects exactly 3 advertising channels.")

# ---------- Tab 4 ----------
with tab4:
    st.markdown("""
**About this project**
- Data: advertising spend on TV, Radio and Newspaper and the resulting sales
- Models compared in the notebook: Linear Regression vs Random Forest. The better one is used here.
- The optimizer tries about 1,600 splits of your total budget and keeps the best prediction
- Honest note: the model shows patterns in past data, not guaranteed cause and effect, and it only knows spend levels it has seen
""")
