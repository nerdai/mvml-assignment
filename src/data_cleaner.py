"""
This Python file does (minimal) data cleaning of the training dataset.
Specifically, it only removes the entries for which "text" is null.
"""

import pandas as pd


def data_cleaner():
    """
    Function responsible for loading `train.csv` and removing records for
    which `text` is nan and storing the resulting DataFrame in a .csv file
    to be used for training.
    """
    raw_df = pd.read_csv("./data/fake_news/train.csv")
    # remove observations with text as nan
    raw_df[~raw_df["text"].isna()].to_csv("./data/fake_news/clean_train.csv")


if __name__ == "__main__":
    data_cleaner()
