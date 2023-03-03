import pandas as pd

def data_loader(train_csv_path: str) -> pd.DataFrame:
    """
    Utility function for loading a csv into a pd.DataFrame

    params:
    [] train_csv_path: str

    returns:
    [] pd.DataFrame
    """
    return pd.read_csv(train_csv_path)
