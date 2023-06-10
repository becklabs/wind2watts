from typing import List
from fitparse import FitFile
import pandas as pd

def extract_all_fit_data(fitfile: FitFile) -> List:
    records = []
    for record in fitfile.get_messages('record'):
        r = {}
        for data in record:
            r[data.name] = data.value
        records.append(r)
    return records

def fit_to_df(fitfile: str) -> pd.DataFrame:
    fitfile = FitFile(fitfile)
    records = extract_all_fit_data(fitfile)
    df = pd.DataFrame(records)
    return df
