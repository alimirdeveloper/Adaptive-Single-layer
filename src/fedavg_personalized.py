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
# ASLA AGGREGATION
# =========================================================
def asla_aggregate(weights_list, losses, energy_scores, sizes):

    eps = 1e-8

    sizes = np.array(sizes, dtype=np.float32)
    losses = np.array(losses, dtype=np.float32)
    energy_scores = np.array(energy_scores, dtype=np.float32)

    # -------------------------
    # NORMALIZATION
    # -------------------------
    data_factor = sizes / (np.sum(sizes) + eps)

    quality_factor = 1.0 / (losses + eps)
    quality_factor = quality_factor / (np.sum(quality_factor) + eps)

    energy_factor = 1.0 / (energy_scores + eps)
    energy_factor = energy_factor / (np.sum(energy_factor) + eps)

    # -------------------------
    # ASLA WEIGHTS
    # -------------------------
    client_weights = (
        0.5 * data_factor +
        0.3 * quality_factor +
        0.2 * energy_factor
    )

    client_weights = client_weights / (np.sum(client_weights) + eps)

    # -------------------------
    # FEDERATED AGGREGATION
    # -------------------------
    new_weights = []

    for layer_weights in zip(*weights_list):

        agg = None

        for i in range(len(weights_list)):

            if agg is None:
                agg = client_weights[i] * layer_weights[i]
            else:
                agg += client_weights[i] * layer_weights[i]

        new_weights.append(agg)

    return new_weights


# =========================================================
# FEDERATED TRAINING (ASLA - 10 CLIENT READY)
# =========================================================
def federated_training(client_data, client_lrs, rounds=5, client_fraction=0.7):

    global_model = build_lstm()
    global_weights = global_model.get_weights()

    snapshots = []
    contributions_log = []

    client_ids = list(client_data.keys())
    assert len(client_ids) > 0, "No clients found"

    for r in range(rounds):

        # -------------------------
        # CLIENT SELECTION
        # -------------------------
        selected_clients = select_clients(client_data, client_fraction)

        # IMPORTANT FIX: handle numpy arrays safely
        if selected_clients is None or len(selected_clients) == 0:
            selected_clients = client_ids
        else:
            selected_clients = list(selected_clients)

        local_weights = []
        losses = []
        energy_scores = []
        sizes = []

        # -------------------------
        # LOCAL TRAINING
        # -------------------------
        for cid in selected_clients:

            X, y = client_data[cid]

            model = local_train(X, y, global_weights, client_lrs[cid])

            loss = model.evaluate(X, y, verbose=0)
            energy = estimate_energy_cost(X)

            weights = get_weights(model)

            # ASLA: update instead of raw weights
            update = [w - gw for w, gw in zip(weights, global_weights)]

            noisy_update = add_dp_noise(update, noise_scale=0.005)

            noisy_weights = [gw + u for gw, u in zip(global_weights, noisy_update)]

            local_weights.append(noisy_weights)
            losses.append(loss)
            energy_scores.append(energy)
            sizes.append(len(X))

        # -------------------------
        # AGGREGATION
        # -------------------------
        global_weights = asla_aggregate(
            local_weights,
            losses,
            energy_scores,
            sizes
        )

        global_model.set_weights(global_weights)

        snapshots.append([w.copy() for w in global_weights])

        # -------------------------
        # CONTRIBUTION LOG (FIXED)
        # -------------------------
        sizes_np = np.array(sizes, dtype=np.float32)
        losses_np = np.array(losses, dtype=np.float32)
        energy_np = np.array(energy_scores, dtype=np.float32)

        eps = 1e-8

        data_factor = sizes_np / (np.sum(sizes_np) + eps)

        quality_factor = 1.0 / (losses_np + eps)
        quality_factor = quality_factor / (np.sum(quality_factor) + eps)

        energy_factor = 1.0 / (energy_np + eps)
        energy_factor = energy_factor / (np.sum(energy_factor) + eps)

        contributions = (
            0.5 * data_factor +
            0.3 * quality_factor +
            0.2 * energy_factor
        )

        contributions = contributions / (np.sum(contributions) + eps)

        # IMPORTANT: store it per round
        full_vector = np.zeros(len(client_ids))
        for i, cid in enumerate(selected_clients):
            idx = client_ids.index(cid)
            full_vector[idx] = contributions[i]

        contributions_log.append(full_vector)

    return global_model, snapshots, contributions_log