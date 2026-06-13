import numpy as np

def asla_aggregate(weights_list, metrics, energy_scores, sizes):

    new_weights = []

    total_clients = len(weights_list)

    # -------------------------
    # compute raw scores
    # -------------------------
    scores = []

    for i in range(total_clients):

        data_factor = sizes[i] / np.sum(sizes)

        quality_factor = 1 / (metrics[i] + 1e-8)

        energy_factor = 1 / (energy_scores[i] + 1e-8)

        score = (
                    0.4 * data_factor +
                    0.4 * quality_factor +
                    0.2 * energy_factor
                )

        scores.append(score)

    # -------------------------
    # NORMALIZATION (CRITICAL)
    # -------------------------
    scores = np.array(scores)
    scores = scores / (np.sum(scores) + 1e-8)

    # -------------------------
    # weighted aggregation
    # -------------------------
    for layer_weights in zip(*weights_list):

        aggregated = None

        for i in range(total_clients):

            if aggregated is None:
                aggregated = scores[i] * layer_weights[i]
            else:
                aggregated += scores[i] * layer_weights[i]

        new_weights.append(aggregated)

    return new_weights