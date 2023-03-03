"""
This file produces the classification metrics report from a supplied labels
as well as predictions data files.
"""

import argparse
import pandas as pd
from sklearn import metrics
from utils.data_loader import data_loader

parser = argparse.ArgumentParser(
    prog='MVML fake news model metrics',
    description='Takes a batch of predictions and its true labels and produces\
a classification report.'
)
parser.add_argument('--preds', type=str, required=True,
                    help='path to preds data csv')
parser.add_argument('--labels', type=str, required=True,
                    help='path to labels data csv')


def produce_report():
    """
    This function loads user specified labels and preds files and produces
    the sklearn metrics.
    """
    # load data
    args = parser.parse_args()
    labels = data_loader(args.labels)
    preds = data_loader(args.preds)

    # metrics
    results = pd.merge(preds, labels, on="id")
    print(metrics.classification_report(results.label, results.pred))


if __name__ == "__main__":
    produce_report()
