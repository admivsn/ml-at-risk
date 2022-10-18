import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report


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

    # Calculate class weight (as slightly imbalanced)
    class_weight = y.value_counts(normalize=True).to_dict()

    # Define model
    estimator = LogisticRegression(
        class_weight=class_weight,
        max_iter=500  # many still fail to converge...
    )

    # Define parameter grid (note not all combinations are valid, so using grid as list of dicts)
    param_grid = [
        {
            "solver": ['newton-cg', 'lbfgs', 'sag'],
            "penalty": ['l2'],
            "C": [100, 10, 1, 0.1, 0.01],
        },
        {
            "solver": ['newton-cg', 'lbfgs', 'sag'],
            "penalty": ['none'],
        },
        {
            "solver": ['liblinear'],
            "penalty": ['l1', 'l2'],
            "C": [100, 10, 1, 0.1, 0.01],
        },
    ]

    # Use brute force 5-fold CV to tune hyperparameters of model
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)
    df_results = pd.DataFrame(
        GridSearchCV(
            estimator=estimator, 
            param_grid=param_grid, 
            n_jobs=-1, 
            cv=cv, 
            scoring='accuracy', 
            error_score=0,
            return_train_score=True
        ).fit(X, y).cv_results_
    )

    # Subset cols
    df_results = df_results[["params", "mean_fit_time", "mean_score_time", "mean_train_score", "std_train_score", "mean_test_score", "std_test_score"]]

    # Here we can decide how we want to rank the models, i.e. by mean_test_score
    # Note: We could assess the model performance based on multiple cols
    df_results = df_results.sort_values(by="mean_test_score", ascending=False).reset_index(drop=True)
    print("CV model results:")
    print(df_results.to_string())

    # Get the best model and params
    best_model = df_results.iloc[0, ]

    # Retrain on entire dataset
    estimator = estimator.set_params(**best_model.params)
    estimator.fit(X, y)
    print("Best model trained on full dataset:")
    print(classification_report(y, estimator.predict(X)))

    # Write model to file
    joblib.dump(estimator, 'app/data/model.joblib')


# run train() when the file is executed
if __name__ == '__main__':
    train()
