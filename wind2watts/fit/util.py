import os
from typing import List, Dict

import pandas as pd
from fitparse import FitFile


def extract_all_fit_data(fitfile: FitFile) -> List[Dict]:
    """
    Returns a list of dictionaries, where each dictionary is a record from the fit file
    """
    records = []
    for record in fitfile.get_messages("record"):
        r = {}
        for data in record:  # type: ignore
            r[data.name] = data.value
        records.append(r)
    return records


def fit_to_df(file_path: str) -> pd.DataFrame:
    """
    Returns a dataframe with all records from the fit file
    """
    fitfile = FitFile(file_path)
    records = extract_all_fit_data(fitfile)
    df = pd.DataFrame(records)
    return df


def clean_dir(path: str) -> None:
    """
    Removes all files in the given directory that
    are not .fit files or do not contain a power column
    """
    for file in os.listdir(path):
        if not file.endswith(".fit"):
            continue
        fit_df = fit_to_df(path + file)
        if "power" not in fit_df.columns:
            print(file)
            os.remove(path + file)
