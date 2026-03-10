import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("Loading dataset...")

df = pd.read_csv("data/processed/final_dataset.csv")

print("Original shape:", df.shape)

# Separate features and label
X = df.drop(columns=["track_id", "genre"])
y = df["genre"]

# Remove columns with NaN
X = X.dropna(axis=1)

# Fill remaining NaN
X = X.fillna(X.mean())

print("Feature shape after cleaning:", X.shape)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Scaling complete")

# PCA dimensionality reduction
pca = PCA(n_components=50)

X_pca = pca.fit_transform(X_scaled)

print("PCA complete")
print("New feature shape:", X_pca.shape)

# Convert back to dataframe
X_pca = pd.DataFrame(X_pca)

final_df = pd.concat([X_pca, y.reset_index(drop=True)], axis=1)

print("Final dataset shape:", final_df.shape)

final_df.to_csv("data/processed/model_dataset.csv", index=False)

print("Saved processed dataset")