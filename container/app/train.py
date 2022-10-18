import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


def preprocess(df):
    """
    Preprocess df for model.
    """
    # Tidy categorical cols for model
    df["bet"] = df["product"] == "bet"
    df = df.drop(columns=["product"])
    df["age"] = df["age"].map({"<30": 0, "30-50": 1, "50+": 2})

    # Engineer useful(?) features
    df["deposits_staked"] = df.apply(lambda x: 0 if x["deposits"] == 0 else x["stakes"] / x["deposits"], axis=1)
    df["deposits_withdrawn"] = df.apply(lambda x: 0 if x["deposits"] == 0 else x["withdrawals"] / x["deposits"], axis=1)
    df["net_withdrawn"] = df["withdrawals"] - df["deposits"]

    return df


def train():
    """
    Train the model.
    """

    df = pd.read_csv('app/data/train.csv')

    # Preprocess training data
    df = preprocess(df)

    # Naively drop rows with missing values
    # TODO: Implement a more robust way to handle these rows (i.e. impute)
    df = df.dropna()

    # Subset data into features and target
    X = df.drop(columns="target")
    y = df["target"]

    # Define model
    estimator = LogisticRegression(penalty="none")

    # Train on entire dataset
    estimator.fit(X, y)

    # Print summary of model performance
    accuracy = accuracy_score(y, estimator.predict(X))
    print(f"Train accuracy: {accuracy:.3f}")

    # Write model to file
    joblib.dump(estimator, 'app/data/model.joblib')


# run train() when the file is executed
if __name__ == '__main__':
    train()
