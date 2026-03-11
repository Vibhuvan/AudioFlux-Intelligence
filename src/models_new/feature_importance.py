import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

print("Loading dataset...")

df = pd.read_csv("data/processed/model_dataset.csv")

X = df.drop(columns=["genre"])

print("Loading trained Random Forest model...")

model = joblib.load("models/randomforest_model.pkl")

importances = model.feature_importances_

indices = np.argsort(importances)[::-1]

top_n = 20

labels = [f"PC{i}" for i in indices[:top_n]]

plt.figure(figsize=(12,6))
plt.title("Top PCA Feature Importances")

plt.bar(range(top_n), importances[indices[:top_n]])

plt.xticks(range(top_n), labels, rotation=90)

plt.tight_layout()

plt.savefig("feature_importance.png")
print("Feature importance saved")