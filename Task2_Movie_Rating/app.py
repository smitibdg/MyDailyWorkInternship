# Movie Rating Predictor - Streamlit app
# Run:  streamlit run app.py   (run the notebook first so movie_model.pkl exists)
import random
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Movie Rating Predictor", page_icon="🎬", layout="wide")

st.markdown("""
<style>
.hero{padding:1.6rem 2rem;border-radius:18px;margin-bottom:1rem;color:white;
      background:linear-gradient(120deg,#1b1b2f 0%,#6a1b4d 60%,#f5b50f 150%);}
.hero h1{margin:0;font-size:2.3rem;color:white}
.hero p{margin:.3rem 0 0 0;opacity:.9}
.card{padding:1rem 1.3rem;border-radius:14px;border:1px solid rgba(128,128,128,.3);background:rgba(106,27,77,.08)}
.big{font-size:3.2rem;font-weight:800;line-height:1.1}
</style>
<div class="hero"><h1>🎬 Will This Movie Be a Hit?</h1>
<p>Pick a cast, a director and a genre. The model predicts the IMDb rating. MyDailyWork Data Science Internship - Task 2</p></div>
""", unsafe_allow_html=True)


@st.cache_resource
def load():
    return joblib.load("movie_model.pkl")


b = load()
NEW = "(new / unknown)"
directors = [NEW] + b["directors"]
actors = [NEW] + b["actors"]

# ---------- Random movie button (sets widget values) ----------
def randomise():
    st.session_state.year = random.randint(b["year_min"], b["year_max"])
    st.session_state.duration = random.randint(90, 180)
    st.session_state.votes = random.choice([200, 1000, 5000, 20000, 100000])
    st.session_state.genres = random.sample(b["top_genres"], 2)
    st.session_state.director = random.choice(b["directors"])
    st.session_state.a1 = random.choice(b["actors"])
    st.session_state.a2 = random.choice(b["actors"])
    st.session_state.a3 = random.choice(b["actors"])


st.sidebar.header("🎞️ Your movie")
st.sidebar.button("🎲 Surprise me", on_click=randomise)
pre_release = st.sidebar.toggle("Pre-release mode (no votes yet)", value=True)
year = st.sidebar.slider("Release year", b["year_min"], b["year_max"], 2015, key="year")
duration = st.sidebar.slider("Duration (minutes)", 60, 220, 135, key="duration")
if not pre_release:
    votes = st.sidebar.select_slider("Votes on IMDb", [10, 50, 200, 1000, 5000, 20000, 100000, 500000],
                                     value=1000, key="votes")
else:
    votes = 1000
genres = st.sidebar.multiselect("Genre(s)", b["top_genres"], default=[b["top_genres"][0]], key="genres")
director = st.sidebar.selectbox("Director", directors, key="director")
a1 = st.sidebar.selectbox("Lead actor", actors, key="a1")
a2 = st.sidebar.selectbox("Actor 2", actors, key="a2")
a3 = st.sidebar.selectbox("Actor 3", actors, key="a3")


def score(col, person):
    """Average rating of this person's past movies (same idea as the notebook)."""
    return float(b["name_maps"][col].get(person, b["average_rating"])) if person != NEW else b["average_rating"]


director_score = score("Director", director)
actor_scores = [score("Actor 1", a1), score("Actor 2", a2), score("Actor 3", a3)]
actors_score = float(np.mean(actor_scores))

row = {"Year": year, "Duration": duration, "Log Votes": np.log1p(votes),
       "Director score": director_score, "Actors score": actors_score}
for g in b["top_genres"]:
    row["G_" + g] = int(g in genres)

if pre_release:
    model, feats = b["model_no_votes"], b["features_no_votes"]
else:
    model, feats = b["model"], b["features"]
pred = float(np.clip(model.predict(pd.DataFrame([row])[feats])[0], 1, 10))

if pred >= 7.5:
    verdict = "🏆 Critics' favourite"
elif pred >= 6.5:
    verdict = "👍 Solid watch"
elif pred >= 5.5:
    verdict = "😐 Mixed reviews"
else:
    verdict = "📉 Box-office trouble"

tab1, tab2, tab3 = st.tabs(["⭐ Prediction", "🧠 What drives ratings", "ℹ️ About"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=pred, number={"suffix": " / 10"}, title={"text": "Predicted IMDb rating"},
            gauge={"axis": {"range": [1, 10]}, "bar": {"color": "#6a1b4d"},
                   "steps": [{"range": [1, 5.5], "color": "#f4c7c3"}, {"range": [5.5, 6.5], "color": "#fde9b8"},
                             {"range": [6.5, 10], "color": "#c6e8c0"}]}))
        fig.update_layout(height=300, margin=dict(t=60, b=10, l=30, r=30), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig)
    with c2:
        st.markdown(f"""<div class="card"><div class="big">{'⭐' * int(round(pred / 2))}</div>
        <h3>{verdict}</h3>
        <p>Average rating in the dataset: <b>{b['average_rating']:.2f}</b>
        ({pred - b['average_rating']:+.2f} for this movie)</p></div>""", unsafe_allow_html=True)
        st.caption("Pre-release mode ignores votes, because votes only exist after a movie is out."
                   if pre_release else "Votes are included, so this works best for movies already released.")

    st.subheader("🤝 Track record of your team")
    labels = ["Director", "Lead actor", "Actor 2", "Actor 3"]
    values = [director_score] + actor_scores
    bar = go.Figure(go.Bar(x=labels, y=values, marker_color="#6a1b4d",
                           text=[f"{v:.2f}" for v in values], textposition="outside"))
    bar.add_hline(y=b["average_rating"], line_dash="dash", annotation_text="average movie")
    bar.update_layout(height=320, yaxis=dict(range=[0, 10], title="Avg rating of their past movies"),
                      margin=dict(t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(bar)

with tab2:
    st.subheader("Biggest factors in the model")
    imp = pd.Series(model.feature_importances_, index=feats).sort_values().tail(8)
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color="#6a1b4d"))
    fig.update_layout(height=380, margin=dict(t=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig)

with tab3:
    st.markdown("""
**About this project**
- Data: IMDb Movies India (year, duration, genre, director, actors, votes, rating)
- Each director and actor became one number: the average rating of their other movies
- Model: Random Forest, compared with Linear Regression and a "guess the average" baseline
- Two models: one with votes, one without (pre-release mode)
- Honest note: ratings depend on much more than this data, so treat predictions as a rough guide
""")
