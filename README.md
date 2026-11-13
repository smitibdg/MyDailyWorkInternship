# 🚀 MYDAILYWORK - Data Science Internship Projects

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Apps-ff4b4b?logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-f37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Tasks%20completed-5%2F5-brightgreen)

Five beginner-friendly machine learning projects built during my **Data Science Internship at [MyDailyWork](https://www.mydailywork.com)**.
Every project has a **step-by-step Jupyter notebook** (one small step per cell) and an **interactive Streamlit app** so you can play with the model.

---

## 📑 Table of Contents
1. [Projects at a glance](#-projects-at-a-glance)
2. [Demo videos](#-demo-videos)
3. [Repository structure](#-repository-structure)
4. [Getting started](#-getting-started)
5. [Project details](#-project-details)
6. [Tech stack](#-tech-stack)
7. [Key learnings](#-key-learnings)
8. [Author](#-author)

---

## 📊 Projects at a glance

| # | Project | Type | Dataset | Models compared | Key result |
|---|---------|------|---------|-----------------|------------|
| 1 | [Titanic Survival Prediction](Task1_Titanic) | Classification | Titanic passengers (418) | Naive Bayes, Logistic Regression, Random Forest | Accuracy: **XX%** |
| 2 | [Movie Rating Prediction](Task2_Movie_Rating) | Regression | IMDb Movies India | Linear Regression, Random Forest | R²: **X.XX**, MAE: **X.XX** |
| 3 | [Iris Flower Classification](Task3_Iris) | Classification | Iris (150 flowers) | KNN, Decision Tree, Logistic Regression, Random Forest | CV accuracy: **XX%** |
| 4 | [Sales Prediction](Task4_Sales) | Regression | Advertising (TV, Radio, Newspaper) | Linear Regression, Random Forest | R²: **X.XX** |
| 5 | [Credit Card Fraud Detection](Task5_Fraud) | Imbalanced classification | Credit card transactions | Logistic Regression and Random Forest with class weights, under- and oversampling | Recall: **XX%**, Precision: **XX%** |

> 💡 Run each notebook and replace the `XX` values above with your own results.

---

## 🎥 Demo videos

| Task | Video |
|------|-------|
| 1 - Titanic | [▶ Watch on LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7526963211229609985/) |
| 2 - Movie Rating | [▶ Watch on LinkedIn](PASTE_LINKEDIN_URL_2) |
| 3 - Iris | [▶ Watch on LinkedIn](PASTE_LINKEDIN_URL_3) |
| 4 - Sales | [▶ Watch on LinkedIn](PASTE_LINKEDIN_URL_4) |
| 5 - Fraud Detection | [▶ Watch on LinkedIn](PASTE_LINKEDIN_URL_5) |

---

## 📁 Repository structure

```
MYDAILYWORK/
├── README.md
├── requirements.txt
├── .gitignore
│
├── Task1_Titanic/
│   ├── Titanic_Survival_Prediction.ipynb
│   ├── app.py
│   └── Titanic_Survival_Prediction.csv
│
├── Task2_Movie_Rating/
│   ├── Movie_Rating_Prediction.ipynb
│   ├── IMDb Movies India.csv
│   └── app.py
│
├── Task3_Iris/
│   ├── Iris_Flower_Classification.ipynb
│   ├── app.py
│   └── iris.csv
│
├── Task4_Sales/
│   ├── Sales_Prediction.ipynb
│   ├── advertising.csv
│   └── app.py
│
└── Task5_Fraud/
    ├── Credit_Card_Fraud_Detection.ipynb
    ├── creditcard.csv
    └── app.py
```

---

## ⚙️ Getting started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/MYDAILYWORK.git
cd MYDAILYWORK
```

### 2. Install the libraries
```bash
pip install -r requirements.txt
```

### 3. Download the datasets
Put each file inside its task folder.

| Task | File name | Where to get it |
|------|-----------|-----------------|
| 1 | `Titanic_Survival_Prediction.csv` | Included in this repo |
| 2 | `IMDb Movies India.csv` | Included in this repo |
| 3 | `iris.csv` | Included. The notebook also creates it from scikit-learn |
| 4 | `Advertising.csv` | Included in this repo |
| 5 | `creditcard.csv` | Included in this repo |

> ⚠️ `creditcard.csv` is about 150 MB, which is above GitHub's 100 MB limit, so it is listed in `.gitignore`. Download it from Kaggle yourself.

### 4. Run a project
```bash
cd Task1_Titanic
jupyter notebook
```
1. Open the `.ipynb` file and choose **Kernel → Restart & Run All**. This trains the model and saves a `.pkl` file next to the notebook.
2. Start the app in the same folder:
```bash
streamlit run app.py
```
The app opens at `http://localhost:8501`.

**Using Google Colab instead?** Upload the dataset, change the file path to `/content/<file>.csv`, run the notebook, then download the saved `.pkl` file (and `demo_transactions.csv` for Task 5) into your project folder to run the app locally.

---

## 🔍 Project details

### 1️⃣ Titanic Survival Prediction
**Goal:** predict whether a passenger survived.
- Explored survival by gender and ticket class with charts
- Filled missing ages, fares and ports, and turned text columns into numbers
- Compared Naive Bayes, Logistic Regression and Random Forest
- Evaluated with accuracy, precision, recall, F1 and a confusion matrix
- **Insight:** in this 418-passenger file every woman survived and every man died, so accuracy is about 100%. On the larger original Titanic dataset the same approach gives roughly 80%.
- **App:** buy a virtual ticket, see your survival chance and play with "what if" scenarios

### 2️⃣ Movie Rating Prediction
**Goal:** predict a movie's IMDb rating from genre, director, actors, duration, year and votes.
- Cleaned messy text such as `(2019)`, `109 min` and `1,086`
- Turned each director and actor into one number: the average rating of their *other* movies (the movie's own rating is left out so the model cannot cheat)
- Compared Linear Regression, Random Forest and a "guess the average" baseline using R², MAE and RMSE
- Trained a second model **without votes**, because votes only exist after release
- **App:** pick a cast and director, get a predicted rating with a pre-release mode and a 🎲 random movie button

### 3️⃣ Iris Flower Classification
**Goal:** identify the species (setosa, versicolor, virginica) from four measurements.
- Explored the data with a pairplot and boxplots
- Compared KNN, Decision Tree, Logistic Regression and Random Forest with a test split **and** 5-fold cross validation (the test set has only 30 flowers)
- Experiment: how accurate is a model that sees just **one** measurement?
- **App:** sliders for the measurements, a live 2D and 3D chart of where your flower sits, and a 🎲 real-flower button

### 4️⃣ Sales Prediction
**Goal:** predict sales from TV, Radio and Newspaper advertising spend.
- Scatter plots and a correlation heatmap for every channel
- Compared Linear Regression with Random Forest (R², MAE, RMSE) and used the better one
- Business insight: how many extra sales does +10 units of spend on each channel bring?
- **App:** predicted sales, channel response curves and a **budget optimizer** that searches the best split of a total budget

### 5️⃣ Credit Card Fraud Detection
**Goal:** find the few fraudulent transactions among hundreds of thousands of genuine ones.
- Showed why accuracy is misleading when fraud is under 0.2% of the data
- Split the data first, then scaled `Time` and `Amount`, and balanced **only the training set**
- Compared 6 strategies: plain, class weights, undersampling and oversampling with Logistic Regression, plus Random Forest variants
- Judged models with precision, recall and F1 instead of accuracy
- **App:** scan transactions and move an alert-threshold slider to see the trade-off between missed fraud and false alarms live

---

## 🛠 Tech stack
- **Language:** Python
- **Data:** pandas, NumPy
- **Machine learning:** scikit-learn
- **Charts:** Matplotlib, Seaborn, Plotly
- **Apps:** Streamlit
- **Tools:** Jupyter Notebook, Google Colab, Git and GitHub

---

## 🎓 Key learnings
- A **perfect score is a question, not a victory** (Titanic)
- Never let the model see information it would not have in real life, such as votes before a movie releases or a person's own rating (Movie Rating)
- A tiny test set is unreliable, so use **cross validation** (Iris)
- A model shows **patterns, not proof of cause and effect**. Test before moving real money (Sales)
- With rare events, **accuracy lies**. Use precision and recall, and treat the alert threshold as a business decision (Fraud)

---

## 👤 Author
**Your Name**
- LinkedIn: [linkedin.com/in/your-profile](https://www.linkedin.com/in/smitibdg/)
- GitHub: [@your-username](https://github.com/smitibdg)
- Email: smiitibaadugu@gmail.com

🙏 Thanks to **[MyDailyWork](https://www.mydailywork.com)** for the internship and the project ideas.

---

⭐ If this repository helped you, please give it a star!
