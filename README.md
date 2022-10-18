# Data Scientist Interview Task

This repo contains a very simple Machine Learning pipeline using **fake** data.
The training data contains some features and a binary target indicating whether a customer has been deemed at-risk (again, the data is totally fake).
Your task is to look over the code and make improvements.
This could include changes to, for example, the model and prediction server (see below).

We want you to demonstrate your ML Engineering skills in developing product-ready ML models.
Please spend a couple of hours on this task.
If you have any ideas that you don't have time to implement please list then below as we may want to discuss them in your interview.
These notes will help us understand how you think about and approach Data Science problems.
Feel free to also include any other files that show your working / the things you tried.

## Candidate notes

### Things you changed
* Load model once on startup instead of every time `predict` endpoint is called
* Made API input more robust - changed to a list of dicts (each record has named parameters) instead of dict with a single key containing a list of lists (each record has unnamed parameters)
* Use all features in the model
* Engineered some new features to use in the model
* Use `classification_report` to inspect more model metrics
* Tweak model to use parameter for imbalanced data
* Perform hyperparameter tuning on model using brute force 5-fold cross validation
* Add functionality to train multiple models and compare

### Any other ideas or notes
* Current model performs badly for positives in the target class, this needs to be addressed (is there enough variability in the data?)
* Decide if accuracy is the best measure to assess model performance in this cases
* Add unit tests
* Add GH actions for `pep8` and unit tests, and document how to enforce using `pre-commit`
* Add more pre-processing to enhance the model and use `Pipeline` to ensure it is applied identically to real data:
    - Handle missing values better by imputing
    - Scale values 
* Add logging to training script

---

## Using this repo

### Setup

You might want to create a Python environment to manage dependencies.
You can do this with `conda` using the code below.

```
conda create -n example python=3.9
conda activate example
pip install -r container/requirements.txt
```

### Container

Below is the structure of the model container.
If you aren't familiar with Docker you can just ignore the Dockerfile.
The scripts described below abstract away the Docker stuff.
The `app` container the python code to train (`train.py`) the model and the code to create an API for generating predictions from the model (`predict.py`).

```
container/
├── Dockerfile
├── app
│   ├── __init__.py
│   ├── data
│   │   ├── model.joblib
│   │   └── train.csv
│   ├── predict.py
│   ├── train.py
│   └── trained
└── requirements.txt
```

### Scripts

A number of scripts are provided to make it easier for you to work with this repo.

#### Build the container

When you change the code in the container this script will rebuild it for you.

```
./scripts/docker_build.sh
```

#### Train the model

Train the model by running train.py within the container.

```
./scripts/train.sh
```

#### Start the predict server

Start the predict server. This will occupy your terminal window so you can see the logs from the server.

```
./scripts/predict_start_server.sh
```

#### Test the predict server

Test the predict server by passing data to the endpoint created by starting the server.
You'll need to do this is a separate terminal window after running `./scripts/predict_start_server.sh`.

```
python scripts/predict_test_server.py
```