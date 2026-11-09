"""
MYDAILYWORK - Data Science Internship
Task 1: Titanic Survival Prediction

"""

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # saves charts to files (no pop-up windows needed)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)

DATA_FILE = "Titanic_Survival_Prediction.csv"
OUT_DIR = "outputs"
RANDOM_STATE = 42
os.makedirs(OUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")

# 1. Loading the Data
df = pd.read_csv(DATA_FILE)
print("1. Data Overview")
print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values:\n", df.isna().sum()[df.isna().sum() > 0])
print("\nSurvival counts:\n", df["Survived"].value_counts())


# 2. Adding Exploratory Data Analysis (EDA) Charts
def save_eda_charts(data):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    sns.countplot(data=data, x="Survived", ax=axes[0, 0])
    axes[0, 0].set_title("Survival Count (0 = No, 1 = Yes)")

    sns.countplot(data=data, x="Sex", hue="Survived", ax=axes[0, 1])
    axes[0, 1].set_title("Survival by Gender")

    sns.countplot(data=data, x="Pclass", hue="Survived", ax=axes[0, 2])
    axes[0, 2].set_title("Survival by Ticket Class")

    sns.histplot(data=data, x="Age", hue="Survived", bins=25, kde=True,
                 ax=axes[1, 0])
    axes[1, 0].set_title("Age Distribution by Survival")

    sns.boxplot(data=data, x="Pclass", y="Fare", hue="Survived", ax=axes[1, 1])
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Fare by Class (log scale)")

    sns.countplot(data=data, x="Embarked", hue="Survived", ax=axes[1, 2])
    axes[1, 2].set_title("Survival by Port of Embarkation")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/01_eda_charts.png", dpi=130)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.heatmap(data.select_dtypes("number").corr(), annot=True, fmt=".2f",
                cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/02_correlation_heatmap.png", dpi=130)
    plt.close()


save_eda_charts(df)
print("\nSaved EDA charts to outputs/")

# 3. Adding Feature Engineering
def engineer_features(data):
    d = data.copy()

    # Title from name (Mr, Mrs, Miss, Master ...)
    d["Title"] = d["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    d["Title"] = d["Title"].replace({
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
        "Lady": "Rare", "the Countess": "Rare", "Countess": "Rare",
        "Capt": "Rare", "Col": "Rare", "Don": "Rare", "Dona": "Rare",
        "Dr": "Rare", "Major": "Rare", "Rev": "Rare", "Sir": "Rare",
        "Jonkheer": "Rare",
    })

    d["FamilySize"] = d["SibSp"] + d["Parch"] + 1
    d["IsAlone"] = (d["FamilySize"] == 1).astype(int)
    d["HasCabin"] = d["Cabin"].notna().astype(int)
    d["Deck"] = d["Cabin"].str[0].fillna("U")  # U = Unknown
    return d


df_fe = engineer_features(df)

NUM_FEATURES = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone", "HasCabin"]
CAT_FEATURES = ["Pclass", "Sex", "Embarked", "Title", "Deck"]
FEATURES = NUM_FEATURES + CAT_FEATURES

X = df_fe[FEATURES].copy()
X["Pclass"] = X["Pclass"].astype(str)  # treat class as a category
y = df_fe["Survived"]

# 4. Adding the Proprocessing Piplene (impute + scale + one-hot)
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, NUM_FEATURES),
    ("cat", categorical_pipe, CAT_FEATURES),
])

# 5. Adding the Train Test Split Method
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")


# 6. Comparing the Models with Cross-Validation
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

print("2. Model Comparison (5-fold cross-validation on training set)")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = {}
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", clf)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    cv_scores[name] = scores.mean()
    print(f"{name:22s} CV accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

best_name = max(cv_scores, key=cv_scores.get)
print(f"\nBest model by CV: {best_name}")

# 7. The Final Evaluation
best_pipe = Pipeline([("prep", preprocessor), ("model", models[best_name])])
best_pipe.fit(X_train, y_train)
y_pred = best_pipe.predict(X_test)

print(f"3. Test Set Results ({best_name})")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
print("\nClassification report:\n",
      classification_report(y_test, y_pred, target_names=["Died", "Survived"]))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Died", "Survived"]).plot(cmap="Blues")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_confusion_matrix.png", dpi=130)
plt.close()

# 8. Feature Importance
model = best_pipe.named_steps["model"]
feature_names = best_pipe.named_steps["prep"].get_feature_names_out()
if hasattr(model, "feature_importances_"):
    importance = model.feature_importances_
elif hasattr(model, "coef_"):
    importance = np.abs(model.coef_[0])
else:
    importance = None

if importance is not None:
    imp = (pd.Series(importance, index=feature_names)
           .sort_values(ascending=False).head(12))
    plt.figure(figsize=(8, 5))
    sns.barplot(x=imp.values, y=imp.index, color="#1a9100")
    plt.title("Top Features Driving Survival Prediction")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/04_feature_importance.png", dpi=130)
    plt.close()
    print("Top features:\n", imp.head(8))

# 9. Saving the Model
joblib.dump(best_pipe, "titanic_model.pkl")
print("\nSaved model -> titanic_model.pkl")

sample = pd.DataFrame([{
    "Age": 28, "Fare": 70.0, "SibSp": 0, "Parch": 0, "FamilySize": 1,
    "IsAlone": 1, "HasCabin": 1, "Pclass": "1", "Sex": "female",
    "Embarked": "S", "Title": "Miss", "Deck": "C",
}])
prob = best_pipe.predict_proba(sample)[0][1]
print(f"\nDemo passenger (28yo female, 1st class): survival probability = {prob:.2%}")
print("Prediction:", "SURVIVED" if prob >= 0.5 else "DID NOT SURVIVE")
print("\nDone. Check the 'outputs' folder for charts.")