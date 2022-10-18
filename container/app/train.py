import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


estimator = LogisticRegression(
    penalty="none"
)


def train():
    """
    Train the model
    """

    df = pd.read_csv('app/data/train.csv')

    features = df[["stakes", "deposits"]]
    target = np.ravel(df["target"])

    estimator.fit(features, target)

    accuracy = accuracy_score(target, estimator.predict(features))
    print(f"Train accuracy: {accuracy:.3f}")

    joblib.dump(estimator, 'app/data/model.joblib')


# run train() when the file is executed
if __name__ == '__main__':
    train()
