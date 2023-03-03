"""
This file produces the classification metrics report from a supplied labels
as well as predictions data files.
"""

import pandas as pd
from sklearn import metrics
from utils.data_loader import data_loader


def produce_report():
    """
    This function loads user specified labels and preds files and produces
    the sklearn metrics.
    """
    # load data
    labels = data_loader("./data/fake_news/labels.csv")
    preds = data_loader("./preds/predictions.csv")

    # metrics
    results = pd.merge(preds, labels, on="id")
    print(metrics.classification_report(results.label, results.pred))


if __name__ == "__main__":
    produce_report()
