import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================
# METRICS
# =========================
def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# =========================
# 1. GLOBAL MODEL EVALUATION
# =========================
def evaluate_global(model, X_test, y_test):
    preds = model.predict(X_test).flatten()

    return {
        "MAE": mean_absolute_error(y_test, preds),
        "MSE": mean_squared_error(y_test, preds),
        "RMSE": rmse(y_test, preds),
        "MAPE": mape(y_test, preds)
    }


# =========================
# 2. CLIENT-WISE MAPE TABLE (LIKE PAPER)
# =========================
def evaluate_clients_mape(client_data, model_fn):

    """
    Produces:
    Client -> MAPE (like paper Table 3/4 horizontal view)
    """

    results = {}

    for cid, data in client_data.items():

        X_train, y_train = data["train"]
        X_test, y_test = data["val"]

        model = model_fn()
        model.fit(X_train, y_train, epochs=3, verbose=0)

        preds = model.predict(X_test).flatten()

        results[cid] = {
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": rmse(y_test, preds),
            "MAPE": mape(y_test, preds)
        }

    return results


# =========================
# 3. ROUND-WISE FEDERATED TRACKING
# =========================
def track_round_metrics(global_model, X_test, y_test, round_id, history):

    preds = global_model.predict(X_test).flatten()

    history["round"].append(round_id)
    history["MAE"].append(mean_absolute_error(y_test, preds))
    history["RMSE"].append(rmse(y_test, preds))
    history["MAPE"].append(mape(y_test, preds))

    return history


# =========================
# 4. LAYER-WISE AGGREGATION SIMULATION (PAPER TABLE 3/4)
# =========================
def layerwise_evaluation(model, weight_snapshots, X_test, y_test):

    """
    Simulates:
    Layer 1
    Layer 1-2
    Layer 1-3
    Layer 1-4
    """

    results = []

    for depth in range(1, len(weight_snapshots) + 1):

        model.set_weights(weight_snapshots[depth - 1])

        preds = model.predict(X_test).flatten()

        results.append({
            "Layer": depth,
            "MAPE": mape(y_test, preds),
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": rmse(y_test, preds)
        })

    return results


# =========================
# 5. PRINT PAPER TABLES
# =========================
def print_tables(global_metrics, client_metrics, layer_metrics):

    import pandas as pd

    # -------- TABLE 1: GLOBAL --------
    print("\n📊 GLOBAL RESULTS")
    print(pd.DataFrame([global_metrics]))

    # -------- TABLE 2: CLIENTS --------
    print("\n📊 CLIENT MAPE (like paper horizontal table)")
    client_df = pd.DataFrame(client_metrics).T
    print(client_df)

    # -------- TABLE 3/4: LAYERWISE --------
    print("\n📊 LAYER-WISE AGGREGATION (Paper Table 3/4)")
    layer_df = pd.DataFrame(layer_metrics)
    print(layer_df)

    return client_df, layer_df