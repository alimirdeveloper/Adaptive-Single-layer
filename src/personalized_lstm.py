# src/personalized_lstm.py
import numpy as np
import tensorflow as tf
from src.lstm_model import build_lstm


LR_CANDIDATES = [0.05, 0.001, 0.0001]


def select_best_lr(X_train, y_train, X_val, y_val):

    best_model = None
    best_lr = None
    best_loss = float("inf")

    for lr in LR_CANDIDATES:

        model = build_lstm()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="mse"
        )

        model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)

        loss = model.evaluate(X_val, y_val, verbose=0)

        if loss < best_loss:
            best_loss = loss
            best_lr = lr
            best_model = model

    return best_model, best_lr, best_loss


def split_client_data(X, y, clients, sample_clients=10):

    unique = np.unique(clients)

    # take up to 10 clients
    unique = unique[:min(sample_clients, len(unique))]

    result = {}

    for c in unique:

        idx = clients == c
        Xc, yc = X[idx], y[idx]

        if len(Xc) < 50:
            continue  # avoid tiny clients (important for stability)

        split = int(len(Xc) * 0.8)

        result[c] = {
            "train": (Xc[:split], yc[:split]),
            "val": (Xc[split:], yc[split:])
        }

    return result