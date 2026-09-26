"""CSV append that tolerates new columns (rewrites the file with the union schema)."""
import os

import pandas as pd


def append_csv(df, path):
    if not os.path.exists(path):
        df.to_csv(path, index=False)
        return
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if set(df.columns) <= set(header):
        df.reindex(columns=header).to_csv(path, mode="a", header=False, index=False)
    else:
        old = pd.read_csv(path)
        pd.concat([old, df], ignore_index=True).to_csv(path, index=False)
