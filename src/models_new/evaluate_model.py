import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("Loading dataset...")

df = pd.read_csv("data/processed/model_dataset.csv")

X = df.drop(columns=["genre"])
y = df["genre"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Loading Random Forest model...")

model = joblib.load("models/randomforest_model.pkl")

preds = model.predict(X_test)

cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(12,10))

sns.heatmap(cm, cmap="Blues")

plt.title("Genre Classification Confusion Matrix")

plt.savefig("confusion_matrix.png")

print("Confusion matrix saved")