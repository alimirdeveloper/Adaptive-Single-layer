import numpy as np


def compute_client_contributions(weights_list, losses, energy_scores, sizes):

    eps = 1e-8

    sizes = np.array(sizes)
    losses = np.array(losses)
    energy_scores = np.array(energy_scores)

    # normalize components
    data_factor = sizes / (np.sum(sizes) + eps)

    quality_factor = 1.0 / (losses + eps)
    quality_factor = quality_factor / (np.sum(quality_factor) + eps)

    energy_factor = 1.0 / (energy_scores + eps)
    energy_factor = energy_factor / (np.sum(energy_factor) + eps)

    # ASLA final contribution
    contribution = (
        0.5 * data_factor +
        0.3 * quality_factor +
        0.2 * energy_factor
    )

    contribution = contribution / (np.sum(contribution) + eps)

    return contribution