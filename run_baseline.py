# run_baseline.py
from src.load_pjm import run_pipeline
from src.lstm_baseline import train_baseline

(X_train, y_train, _), (X_test, y_test, _), _ = run_pipeline(
    "data/hrl_load_metered.csv"
)

model, _ = train_baseline(X_train, y_train, X_test, y_test)
model.save("baseline.h5")