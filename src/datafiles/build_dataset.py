import pandas as pd

print("Loading features...")

# features has multi-level columns
features = pd.read_csv("data/raw/features.csv", header=[0,1,2], index_col=0)

# flatten column names
features.columns = ['_'.join(col).strip() for col in features.columns.values]

# bring track_id back as column
features = features.reset_index()
features.rename(columns={"index": "track_id"}, inplace=True)

print("Features shape:", features.shape)

print("Loading tracks metadata...")

tracks = pd.read_csv("data/raw/tracks.csv", header=[0,1], index_col=0)

# flatten columns
tracks.columns = ['_'.join(col).strip() for col in tracks.columns.values]

tracks = tracks.reset_index()
tracks.rename(columns={"index": "track_id"}, inplace=True)

print("Tracks shape:", tracks.shape)

# get genre column
genres = tracks[['track_id', 'track_genre_top']].copy()
genres.columns = ['track_id', 'genre']

print("Merging datasets...")

df = pd.merge(features, genres, on="track_id")

df = df.dropna(subset=["genre"])

print("Final dataset shape:", df.shape)

df.to_csv("data/processed/final_dataset.csv", index=False)

print("Dataset saved to data/processed/final_dataset.csv")