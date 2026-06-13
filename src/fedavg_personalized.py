import numpy as np
import tensorflow as tf
from src.lstm_model import build_lstm
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
# ASLA CORE AGGREGATION (PAPER-STYLE)
# =========================================================
def asla_aggregate(weights_list, losses, energy_scores, sizes):

    eps = 1e-8

    sizes = np.array(sizes)
    losses = np.array(losses)
    energy_scores = np.array(energy_scores)

    # -----------------------------
    # NORMALIZED COMPONENTS
    # -----------------------------
    data_factor = sizes / (np.sum(sizes) + eps)

    quality_factor = 1.0 / (losses + eps)
    quality_factor = quality_factor / (np.sum(quality_factor) + eps)

    energy_factor = 1.0 / (energy_scores + eps)
    energy_factor = energy_factor / (np.sum(energy_factor) + eps)

    # -----------------------------
    # FINAL ASLA WEIGHT (CONTROLLED)
    # -----------------------------
    weights = (
        0.5 * data_factor +
        0.3 * quality_factor +
        0.2 * energy_factor
    )

    weights = weights / (np.sum(weights) + eps)

    # -----------------------------
    # AGGREGATION
    # -----------------------------
    new_weights = []

    for layer_weights in zip(*weights_list):

        agg = None

        for i in range(len(weights_list)):

            if agg is None:
                agg = weights[i] * layer_weights[i]
            else:
                agg += weights[i] * layer_weights[i]

        new_weights.append(agg)

    return new_weights


# =========================================================
# FEDERATED TRAINING (ASLA FINAL VERSION)
# =========================================================
def federated_training(client_data, client_lrs, rounds=5, client_fraction=0.7):

    global_model = build_lstm()
    global_weights = global_model.get_weights()

    snapshots = []

    for r in range(rounds):

        selected_clients = select_clients(client_data, client_fraction)

        local_weights = []
        losses = []
        energy_scores = []
        sizes = []

        # -----------------------------
        # CLIENT DROP-OUT SIMULATION
        # -----------------------------
        if len(selected_clients) == 0:
            continue

        for cid in selected_clients:

            X, y = client_data[cid]

            model = local_train(X, y, global_weights, client_lrs[cid])

            loss = model.evaluate(X, y, verbose=0)

            energy = estimate_energy_cost(X)

            weights = get_weights(model)

            # privacy noise (light, realistic)
            noisy_weights = add_dp_noise(weights, noise_scale=0.005)

            local_weights.append(noisy_weights)
            losses.append(loss)
            energy_scores.append(energy)
            sizes.append(len(X))

        # -----------------------------
        # ASLA AGGREGATION
        # -----------------------------
        global_weights = asla_aggregate(
            local_weights,
            losses,
            energy_scores,
            sizes
        )

        global_model.set_weights(global_weights)

        # -----------------------------
        # SNAPSHOT (REAL PAPER STYLE)
        # -----------------------------
        snapshots.append([w.copy() for w in global_weights])

    return global_model, snapshots