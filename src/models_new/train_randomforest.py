import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


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

print("Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)

print("Accuracy:", acc)

print("\nClassification Report:\n")
print(classification_report(y_test, preds))

joblib.dump(model, "models/randomforest_model.pkl")

print("Model saved")