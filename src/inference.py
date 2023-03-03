"""
This Python file is responsible for performing batch inference on new data
observations.

The predictions are saved as .csv in local.
"""

import pandas as pd
from utils.data_loader import data_loader
from joblib import load
from sklearn import metrics


def inference():
    """
    Function responsible for training the ML model and storing the artifact
    binary for later use.
    """
    # load data
    df = pd.read_csv("./data/fake_news/test.csv")
    labels = pd.read_csv("./data/fake_news/labels.csv")

    # some cleaning
    df_clean = df[~df["text"].isna()]
    df_dirty = df[df["text"].isna()]

    # load model
    model = load('model.joblib')

    # run model
    preds = model.predict(df_clean.text)

    # metrics
    df_clean = pd.merge(df_clean, labels, on="id")
    print(metrics.classification_report(df_clean.label, preds))

    # save preds
    df_clean["pred"] = preds
    df_clean[["id", "pred"]].to_csv("./preds/predictions.csv", index=False)


if __name__ == "__main__":
    inference()
