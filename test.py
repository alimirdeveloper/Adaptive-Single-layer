import pandas as pd

df = pd.read_csv("data/hrl_load_metered.csv")
print(df.columns)
print(df.head())