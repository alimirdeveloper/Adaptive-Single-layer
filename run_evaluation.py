from src.load_pjm import run_pipeline
from src.lstm_model import build_lstm
from src.fedavg_personalized import federated_training
from src.personalized_lstm import split_client_data, select_best_lr
import numpy as np
from src.paper_results_engine import (
    evaluate_global,
    client_mape_table,
    layerwise_table,
    print_paper_results
)
import matplotlib.pyplot as plt
# =========================================================
# 1. LOAD DATA
# =========================================================
(X_train, y_train, c_train), (X_test, y_test, c_test), _ = run_pipeline(
    "data/hrl_load_metered.csv"
)

# =========================================================
# 2. BASELINE (CENTRALIZED MODEL)
# =========================================================
central_model = build_lstm()
central_model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=0)

central_metrics = evaluate_global(
    central_model, X_test, y_test, "Centralized"
)

# =========================================================
# 3. CLIENT PREPARATION
# =========================================================
clients = split_client_data(X_train, y_train, c_train)

client_lrs = {}
client_data = {}

for cid, d in clients.items():
    Xtr, ytr = d["train"]
    Xv, yv = d["val"]

    _, lr, _ = select_best_lr(Xtr, ytr, Xv, yv)

    client_lrs[cid] = lr
    client_data[cid] = (Xtr, ytr)

# =========================================================
# 4. FEDERATED TRAINING (ASLA / FEDAVG CORE)
# =========================================================
fed_model, snapshots, contrib_log = federated_training(
    client_data,
    client_lrs
)
fed_metrics = evaluate_global(
    fed_model, X_test, y_test, "FedAvg/ASLA"
)

# =========================================================
# 5. CLIENT-LEVEL TABLE (PAPER TABLE 2 STYLE)
# =========================================================
client_df = client_mape_table(clients, build_lstm)

# =========================================================
# 6. LAYER-WISE TABLE (PAPER TABLE 3/4)
# =========================================================
layer_df = layerwise_table(
    central_model,
    snapshots,
    X_test,
    y_test
)

# =========================================================
# 7. FINAL PAPER OUTPUT
# =========================================================
global_results = [
    central_metrics,
    fed_metrics
]

print_paper_results(
    global_results,
    client_df,
    layer_df
)

clients = list(client_data.keys())

if len(contrib_log) == 0:
    raise ValueError("Contribution log is empty — check federated_training loop")

final_contrib_raw = contrib_log[-1]

# =====================================================
# ALIGN CONTRIBUTIONS TO ALL CLIENTS
# =====================================================
client_index = {c: i for i, c in enumerate(clients)}

final_contrib = np.zeros(len(clients))

# selected clients only
selected_clients = list(client_data.keys())[:len(final_contrib_raw)]

for i, cid in enumerate(selected_clients):
    final_contrib[client_index[cid]] = final_contrib_raw[i]

# =====================================================
# PLOT
# =====================================================
plt.figure(figsize=(10,5))

plt.bar(clients, final_contrib)

plt.title("ASLA Client Contribution (Final Round)")
plt.ylabel("Contribution Weight")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("outputs/figures/asla_client_contribution.png", dpi=300)
plt.close()