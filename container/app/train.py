import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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

    # Define models to try (note for param_grids not all combinations are valid, so using list of valid dicts)
    models = {
        "logistic_regression": {
            "estimator": LogisticRegression(
                class_weight=class_weight,
                max_iter=500  # many still fail to converge...
            ),
            "param_grids": [
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
        },
        "decision_tree": {
            "estimator": DecisionTreeClassifier(
                class_weight=class_weight
            ),
            "param_grids": {
                'max_depth': [2, 3, 5, 10, 20],
                'min_samples_leaf': [5, 10, 20, 50],
                'min_samples_split': [2, 5, 10, 20, 50],
                'criterion': ["gini", "entropy"],
                'splitter': ['best', 'random']
            }
        },
    }

    # Use brute force 5-fold CV to tune hyperparameters of each model and combine results into dataframe
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)
    df_results = pd.concat(
        [
            pd.DataFrame(
                {
                    "name": n,
                    "estimator": m["estimator"],
                    **GridSearchCV(
                        estimator=m["estimator"], 
                        param_grid=m["param_grids"], 
                        n_jobs=-1, 
                        cv=cv, 
                        scoring='accuracy', 
                        error_score=0,
                        return_train_score=True
                    ).fit(X, y).cv_results_
                }
            )
            for n, m in models.items()
        ] 
    )

    # Subset cols
    df_results = df_results[["name", "estimator", "params", "mean_fit_time", "mean_score_time", "mean_train_score", "std_train_score", "mean_test_score", "std_test_score"]]

    # Here we can decide how we want to rank the models, i.e. by mean_test_score
    # Note: We could assess the model performance based on multiple cols
    df_results = df_results.sort_values(by="mean_test_score", ascending=False).reset_index(drop=True)
    print("CV model results:")
    print(df_results.to_string())

    # Get the best model and params
    best_model = df_results.iloc[0, ]

    # Retrain on entire dataset
    estimator = best_model.estimator.set_params(**best_model.params)
    estimator.fit(X, y)
    print("Best model trained on full dataset:")
    print(classification_report(y, estimator.predict(X)))

    # Write model to file
    joblib.dump(estimator, 'app/data/model.joblib')


# run train() when the file is executed
if __name__ == '__main__':
    train()
