# run_federated.py
from src.load_pjm import run_pipeline
from src.personalized_lstm import split_client_data, select_best_lr
from src.fedavg_personalized import federated_training


(X_train, y_train, c_train), _, _ = run_pipeline(
    "data/hrl_load_metered.csv"
)

clients = split_client_data(X_train, y_train, c_train)

client_lrs = {}
client_data = {}

for cid, d in clients.items():

    Xtr, ytr = d["train"]
    Xv, yv = d["val"]

    _, lr, _ = select_best_lr(Xtr, ytr, Xv, yv)

    client_lrs[cid] = lr
    client_data[cid] = (Xtr, ytr)

model, snapshots = federated_training(client_data, client_lrs)
model.save("fedavg.h5")