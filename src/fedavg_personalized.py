import numpy as np
import tensorflow as tf
from src.lstm_model import build_lstm
from src.asla_aggregation import asla_aggregate
from src.personalized_lstm import select_best_lr
from src.client_selection import select_clients
from src.client_profile import estimate_energy_cost
from src.privacy import add_dp_noise


# =========================================================
# UTILITIES
# =========================================================
def get_weights(model):
    return model.get_weights()


def set_weights(model, weights):
    model.set_weights(weights)


# =========================================================
# LOCAL TRAINING
# =========================================================
def local_train(X, y, global_weights, lr):

    model = build_lstm()
    model.set_weights(global_weights)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse"
    )

    model.fit(X, y, epochs=3, batch_size=32, verbose=0)

    return model


# =========================================================
# ASLA / FEDAVG CORE TRAINING
# =========================================================
def federated_training(client_data, client_lrs, rounds=5, client_fraction=0.7):

    global_model = build_lstm()
    global_weights = global_model.get_weights()

    snapshots = []   # IMPORTANT for paper Table 3/4

    # =====================================================
    # FEDERATED ROUNDS
    # =====================================================
    for r in range(rounds):

        local_weights = []
        client_metrics = []
        energy_scores = []
        sizes = []

        # -------------------------------------------------
        # CLIENT SELECTION (ASLA FEATURE)
        # -------------------------------------------------
        selected_clients = select_clients(client_data, fraction=client_fraction)

        # -------------------------------------------------
        # LOCAL TRAINING
        # -------------------------------------------------
        for cid in selected_clients:

            X, y = client_data[cid]

            model = local_train(X, y, global_weights, client_lrs[cid])

            # validation proxy (quality)
            loss = np.log(model.evaluate(X, y, verbose=0) + 1e-8)

            # energy proxy
            energy = estimate_energy_cost(X)

            # weights
            weights = get_weights(model)

            # privacy layer (DP noise)
            noisy_weights = add_dp_noise(weights)

            local_weights.append(noisy_weights)
            client_metrics.append(loss)
            energy_scores.append(energy)
            sizes.append(len(X))

        # -------------------------------------------------
        # ASLA AGGREGATION
        # -------------------------------------------------
        global_weights = asla_aggregate(
            local_weights,
            client_metrics,
            energy_scores,
            sizes
        )

        global_model.set_weights(global_weights)

        # -------------------------------------------------
        # SNAPSHOT (CRITICAL FOR PAPER TABLE 3/4)
        # -------------------------------------------------
        snapshots.append([w.copy() for w in global_weights])

    return global_model, snapshots