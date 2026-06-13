# src/plotting.py
import os
import matplotlib.pyplot as plt


# ---------------------------
# Ensure output folder exists
# ---------------------------
def ensure_dir(path="outputs/figures"):
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------
# Figure 1: Model comparison
# ---------------------------
def plot_model_comparison(metrics, centralized, fedavg, personalized, save_dir="outputs/figures"):

    save_dir = ensure_dir(save_dir)

    x = range(len(metrics))

    plt.figure(figsize=(8, 5))

    plt.bar([i - 0.2 for i in x], centralized, width=0.2, label="Centralized")
    plt.bar(x, fedavg, width=0.2, label="FedAvg")
    plt.bar([i + 0.2 for i in x], personalized, width=0.2, label="Personalized")

    plt.xticks(x, metrics)
    plt.title("Model Performance Comparison")
    plt.legend()

    path = os.path.join(save_dir, "figure_1_model_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    return path


# ---------------------------
# Figure 2: Improvement
# ---------------------------
def plot_improvement(metrics, improvement, save_dir="outputs/figures"):

    save_dir = ensure_dir(save_dir)

    plt.figure(figsize=(6, 4))

    plt.bar(metrics, improvement)

    plt.title("FedAvg Improvement over Centralized (%)")

    path = os.path.join(save_dir, "figure_2_improvement.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    return path


# ---------------------------
# Figure 3: RMSE comparison
# ---------------------------
def plot_rmse(models, rmse_values, save_dir="outputs/figures"):

    save_dir = ensure_dir(save_dir)

    plt.figure(figsize=(6, 4))

    plt.bar(models, rmse_values)

    plt.title("RMSE Comparison Across Models")

    path = os.path.join(save_dir, "figure_3_rmse.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    return path