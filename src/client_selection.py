import numpy as np

def select_clients(client_data, fraction=0.6):

    clients = list(client_data.keys())
    k = max(1, int(len(clients) * fraction))

    return np.random.choice(clients, k, replace=False)