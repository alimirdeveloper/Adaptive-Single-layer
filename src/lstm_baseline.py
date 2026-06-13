# src/lstm_baseline.py
from src.lstm_model import build_lstm


def train_baseline(X_train, y_train, X_test, y_test):

    model = build_lstm()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=64,
        verbose=1
    )

    return model, history