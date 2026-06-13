# src/load_pjm.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

SEQUENCE_LENGTH = 24


def load_data(path):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime_beginning_utc"])
    df = df.sort_values(["load_area", "datetime"])
    return df


def select_target(df):
    return df[["load_area", "datetime", "mw"]]


def scale_per_client(df):
    scalers = {}
    scaled = []

    for c in df["load_area"].unique():
        temp = df[df["load_area"] == c].copy()
        scaler = StandardScaler()
        temp["mw"] = scaler.fit_transform(temp[["mw"]])
        scalers[c] = scaler
        scaled.append(temp)

    return pd.concat(scaled), scalers


def create_sequences(df, seq_len=24):
    X, y, clients = [], [], []

    for c in df["load_area"].unique():
        temp = df[df["load_area"] == c].sort_values("datetime")
        values = temp["mw"].values

        for i in range(len(values) - seq_len - 1):
            X.append(values[i:i+seq_len])
            y.append(values[i+seq_len])
            clients.append(c)

    X = np.array(X).reshape(-1, seq_len, 1)
    y = np.array(y)

    return X, y, np.array(clients)


def split_data(X, y, clients, split=0.8):
    idx = int(len(X) * split)

    return (
        X[:idx], y[:idx], clients[:idx],
        X[idx:], y[idx:], clients[idx:]
    )


def run_pipeline(path):

    df = load_data(path)
    df = select_target(df)

    df, scalers = scale_per_client(df)

    X, y, clients = create_sequences(df)

    X_train, y_train, c_train, X_test, y_test, c_test = split_data(X, y, clients)

    return (
        (X_train, y_train, c_train),
        (X_test, y_test, c_test),
        scalers
    )