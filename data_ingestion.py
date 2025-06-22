import pandas as pd

def load_data(path="data/wafer.csv"):
    df = pd.read_csv(path)
    return df
