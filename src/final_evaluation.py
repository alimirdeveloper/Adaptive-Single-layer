# src/final_evaluation.py
import numpy as np
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_metrics(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)

    return mae, mse, rmse


def evaluate_model(model, X, y):

    preds = model.predict(X).flatten()
    return compute_metrics(y, preds)


# ---------------------------
# ✅ FIXED CLIENT EVALUATION
# ---------------------------
def evaluate_clients(client_data, model_fn):

    maes, mses, rmses = [], [], []

    for _, data in client_data.items():

        # ✔ CORRECT SPLIT USAGE
        X_train, y_train = data["train"]
        X_test, y_test = data["val"]

        # train local models
        model = model_fn()
        model.fit(X_train, y_train, epochs=3, verbose=0)

        # evaluate on UNSEEN data
        preds = model.predict(X_test).flatten()

        mae, mse, rmse = compute_metrics(y_test, preds)

        maes.append(mae)
        mses.append(mse)
        rmses.append(rmse)

    return {
        "MAE": float(np.mean(maes)),
        "MSE": float(np.mean(mses)),
        "RMSE": float(np.mean(rmses))
    }


def print_comparison(c, f, p):

    import pandas as pd

    df = pd.DataFrame([
        ["Centralized", c[0], c[1], c[2]],
        ["FedAvg", f[0], f[1], f[2]],
        ["Personalized", p["MAE"], p["MSE"], p["RMSE"]],
    ], columns=["Model", "MAE", "MSE", "RMSE"])

    print("\n📊 FINAL PAPER RESULTS")
    print("======================")
    print(df)

    return df

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100