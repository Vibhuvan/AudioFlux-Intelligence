import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


print("Loading dataset...")

df = pd.read_csv("data/processed/model_dataset.csv")

X = df.drop(columns=["genre"])
y = df["genre"]

print("Dataset shape:", X.shape)


# stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


models = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "RandomForest": RandomForestClassifier(n_estimators=200),
    "GradientBoosting": GradientBoostingClassifier()
}

results = {}

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print("Accuracy:", acc)

    results[name] = acc


best_model_name = max(results, key=results.get)

print("\nBest model:", best_model_name)

best_model = models[best_model_name]

joblib.dump(best_model, "models/best_model.pkl")

print("Model saved to models/best_model.pkl")