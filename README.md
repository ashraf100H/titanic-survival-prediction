# Titanic Survival Prediction

A beginner machine learning project using the Titanic dataset to predict whether a passenger survived the Titanic disaster.

This project was created as part of my machine learning learning journey, with a focus on understanding the complete machine learning workflow from data exploration to model evaluation.

## Project Goal

Can we predict whether a Titanic passenger survived based on passenger information such as class, gender, age, fare, family size, and embarkation port?

## Dataset

The dataset is from the Kaggle Titanic competition.

The training dataset contains 891 passenger records.

The target variable is:

- `Survived`
  - `0` = Did not survive
  - `1` = Survived

## Features Used

The following features were selected:

- `Pclass`
- `Sex`
- `Age`
- `SibSp`
- `Parch`
- `Fare`
- `Embarked`
- `FamilySize`

### Feature Engineering

A new feature called `FamilySize` was created:

`FamilySize = SibSp + Parch + 1`

This represents the passenger's immediate family group size, including the passenger.

## Data Preprocessing

The project included:

- Exploring the dataset
- Identifying missing values
- Removing the `Cabin` feature because of its large number of missing values
- Imputing missing values
- Encoding categorical features
- Creating the `FamilySize` feature
- Removing identifier-like features such as `PassengerId`, `Name`, and the raw `Ticket` value
- Splitting the data into training and testing sets

## Model

A Logistic Regression classifier was used as the baseline model.

The dataset was split into:

- 80% training data
- 20% testing data

A fixed `random_state=42` was used to make the split reproducible.

## Results

The Logistic Regression model achieved:

**Accuracy: 79.89%**

### Classification Results

| Class | Precision | Recall | F1-score |
|------|-----------|--------|----------|
| Did not survive | 0.82 | 0.85 | 0.83 |
| Survived | 0.77 | 0.73 | 0.75 |

The model correctly classified:

- 89 passengers who did not survive
- 54 passengers who survived

It incorrectly classified:

- 16 non-survivors as survivors
- 20 survivors as non-survivors

## Key Findings

The analysis of `FamilySize` showed that survival rates differed considerably between family sizes.

Passengers traveling alone had an observed survival rate of approximately 30.35%, while passengers with a family size of four had an observed survival rate of approximately 72.41%.

However, these are descriptive statistics from the historical dataset and should not be interpreted as model predictions or causal relationships.

The Logistic Regression model performed somewhat better at identifying passengers who did not survive than passengers who survived.

## Limitations

This is an introductory machine learning project and the goal was to understand the end-to-end workflow rather than maximize predictive performance.

The project uses a simple Logistic Regression model and basic feature engineering.

For a production-level workflow, preprocessing operations such as imputation should be fitted using the training data only to prevent data leakage. A Scikit-learn pipeline would be a better approach.

Further improvements could include:

- More advanced feature engineering
- Cross-validation
- Hyperparameter tuning
- Tree-based models such as Random Forest or Gradient Boosting
- Scikit-learn Pipelines

##  Interactive Streamlit Application

The project was extended with an interactive Streamlit application that allows users to enter passenger information and receive a survival prediction.

The application:

1. Accepts passenger information through an interactive form.
2. Converts categorical inputs into the same encoding used during model training.
3. Automatically calculates the `FamilySize` feature.
4. Passes the processed data to the trained Logistic Regression model.
5. Displays the predicted outcome and survival probabilities.
6. Provides a simple coefficient-based explanation of which features influenced the prediction.

The application can be launched locally with:

```bash
streamlit run app.py


## Project Structure

And update your project structure to:

```text
titanic-survival-prediction/
│
├── data/
│   └── train.csv
│
├── model/
│   └── titanic_model.pkl
│
├── notebook/
│   └── titanic_analysis.ipynb
│
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
___-

 

