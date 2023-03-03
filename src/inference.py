"""
This Python file is responsible for performing batch inference on new data
observations.

The predictions are saved as .csv in local.
"""

from joblib import load
from utils.data_loader import data_loader


def inference():
    """
    Function responsible for training the ML model and storing the artifact
    binary for later use.
    """
    # load data
    new_df = data_loader("./data/fake_news/test.csv")

    # some cleaning
    df_clean = new_df[~new_df["text"].isna()]
    df_dirty = new_df[new_df["text"].isna()]

    # load & run model
    model = load('model.joblib')
    preds = model.predict(df_clean.text)

    # save preds
    df_clean["pred"] = preds
    df_clean[["id", "pred"]].to_csv("./preds/predictions.csv", index=False)

    if not df_dirty.empty():
        print(f"{df_dirty.shape[0]} observations had no `text` field and thus \
were not predicted.")


if __name__ == "__main__":
    inference()
