# Task 1 - Titanic Survival Prediction (MyDailyWork Data Science Internship)

Predicts whether a Titanic passenger survived using passenger details
(age, gender, ticket class, fare, cabin, family size, title).

## Files
- `titanic_survival.py` - EDA, feature engineering, model comparison, evaluation, saves model
- `app.py` - Streamlit demo app
- `Titanic_Survival_Prediction.csv` - dataset
- `outputs/` - generated charts

## How to run
```
pip install -r requirements.txt
python titanic_survival.py
streamlit run app.py
```

## Approach
1. Explored the data and plotted survival by gender, class, age, fare, port
2. Engineered features: Title, FamilySize, IsAlone, HasCabin, Deck
3. Imputed missing values, scaled numbers, one-hot encoded categories (sklearn Pipeline)
4. Compared Logistic Regression, Random Forest, Gradient Boosting with 5-fold CV
5. Evaluated on a held-out test set (accuracy, precision, recall, F1, confusion matrix)

## Note
In this particular 418-row file, survival lines up exactly with gender
(all females survived, all males died), so models reach 100% accuracy.
On the original, larger Titanic training data, expect roughly 80-84%.
