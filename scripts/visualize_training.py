# visualize_training.py — Visualize predictions and export results
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# ✅ Create outputs/ folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Load data
with open("data/train_grids.pkl", "rb") as f:
    X = pickle.load(f)
with open("data/train_labels.pkl", "rb") as f:
    y = pickle.load(f)

# Load model (fallback to custom loss if needed)
model = load_model("models/cnn_binding_affinity.h5", compile=False)
model.compile(optimizer="adam", loss=MeanSquaredError(), metrics=["mae"])

# Predict
y_pred = model.predict(X).flatten()

# Save actual vs predicted to CSV
results_df = pd.DataFrame({
    "True_Ki": y,
    "Predicted_Ki": y_pred
})
results_df.to_csv("outputs/predictions_vs_actual.csv", index=False)
print("✅ Saved predictions to outputs/predictions_vs_actual.csv")

# Metrics
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
print(f"RMSE: {rmse:.4f}\n")

# --- Plot 1: Scatter plot (True vs Predicted) ---
plt.figure(figsize=(6, 6))
plt.scatter(y, y_pred, alpha=0.4)
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
plt.xlabel("True Ki")
plt.ylabel("Predicted Ki")
plt.title("True vs Predicted Binding Affinity")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/scatter_true_vs_pred.png")

# --- Plot 2: Histogram of prediction errors ---
errors = y_pred - y
plt.figure(figsize=(6, 4))
plt.hist(errors, bins=30, color='skyblue', edgecolor='black')
plt.xlabel("Prediction Error")
plt.ylabel("Count")
plt.title("Distribution of Prediction Errors")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/hist_prediction_error.png")
