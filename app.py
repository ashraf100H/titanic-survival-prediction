import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model/titanic_model.pkl")

st.title("🚢 Titanic Survival Prediction")

st.write(
    "Enter the passenger information below to predict "
    "the probability of survival."
)

Pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd Class)", options=[1, 2, 3])

Sex = st.selectbox("Sex", options=["male", "female"])

Age = st.slider("Age", min_value=0, max_value=100, value=30)

sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)

parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)

fare = st.number_input("Fare", min_value=0.0, max_value=500.0, value=30.0)

Embarked = st.selectbox("Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)", options=["C", "Q", "S"])

# Convert categorical values to model encoding
sex_encoded = 0 if Sex == "male" else 1

embarked_q = 1 if Embarked == "Q" else 0
embarked_s = 1 if Embarked == "S" else 0


family_size = sibsp + parch + 1

# Prediction button
if st.button("Predict Survival"):

    # Create a DataFrame for the input data
    input_data = pd.DataFrame(
        {
            "Pclass": [Pclass],
            "Sex": [sex_encoded],
            "Age": [Age],
            "SibSp": [sibsp],
            "Parch": [parch],           
            "Fare": [fare],
            "Embarked_Q": [embarked_q],
            "Embarked_S": [embarked_s],
            "FamilySize": [family_size],
        }
    )

    st.subheader("Input Data")
    st.dataframe(input_data)
    # Make the prediction
    prediction = model.predict(input_data)[0]

    # Get prediction probabilities
    probability = model.predict_proba(input_data)[0]

    # Extract survival probability
    survival_probability = probability[1]
    non_survival_probability = probability[0]

    # Display the prediction result
    if prediction == 1:
        st.success(
            f"The passenger is predicted to survive "
            f"with a probability of {survival_probability:.2%}."
        )
    else:
        st.error(
            f"The passenger is predicted not to survive "
            f"with a probability of {non_survival_probability:.2%}."
        )

    # Display both probabilities
    st.write(
        f"**Survival probability:** {survival_probability:.2%}"
    )

    st.write(
        f"**Non-survival probability:** {non_survival_probability:.2%}"
    )

    # -----------------------------------------
    # Model explanation
    # -----------------------------------------

    st.subheader("What influenced this prediction?")

    # Get model coefficients
    coefficients = model.coef_[0]

    # calculate the contribution of each feature to the prediction
    contributions = coefficients * input_data.iloc[0]

    explanation_df = pd.DataFrame(
        {
            "Feature": input_data.columns,
            "Contribution": contributions.values,
        }
    )

   # sort by absolute contribution values
    explanation_df["AbsContribution"] = explanation_df["Contribution"].abs()
    explanation_df = explanation_df.sort_values(by="AbsContribution", ascending=False)

    # Display the most influential features
    for _, row in explanation_df.head(5).iterrows():

        feature = row["Feature"]
        contribution = row["Contribution"]

        if contribution > 0:
            st.write(
                f"**{feature}** → pushed the prediction "
                f"toward survival."
            )
        else:
            st.write(
                f"**{feature}** → pushed the prediction "
                f"toward non-survival."
            )

    st.caption(
        "These explanations describe how the trained model used "
        "the features."
    )

