import numpy as np

def estimate_energy_cost(X):

    variance = np.var(X)
    size_factor = np.log(len(X) + 1)

    energy = variance * size_factor

    # normalize energy (VERY IMPORTANT)
    return float(energy / (np.mean(X) + 1e-8))