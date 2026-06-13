import numpy as np

def add_dp_noise(weights, noise_scale=0.01):

    noisy = []

    for w in weights:
        noisy.append(w + np.random.normal(0, noise_scale, size=w.shape))

    return noisy