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
    clean_train = raw_df[~raw_df["text"].isna()]
    # dedup observations with repeated `(text, label)` pairs
    clean_deduped = pd.concat([
        clean_train[~clean_train[['text', 'label']].duplicated(keep=False)],
        clean_train[clean_train[['text', 'label']].duplicated(keep='last')]
    ], axis=0)
    clean_deduped.to_csv("./data/fake_news/clean_train.csv")


if __name__ == "__main__":
    data_cleaner()
