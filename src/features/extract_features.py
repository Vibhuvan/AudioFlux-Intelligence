import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_PATH = "data/raw/fma_small"
OUTPUT_PATH = "data/processed/features.csv"

features = []

for root, dirs, files in os.walk(DATA_PATH):
    for file in tqdm(files):
        if file.endswith(".mp3"):
            path = os.path.join(root, file)

            try:
                y, sr = librosa.load(path, sr=22050)

                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
                mfcc_mean = np.mean(mfcc, axis=1)

                features.append(mfcc_mean)

            except Exception as e:
                print("error:", path)

df = pd.DataFrame(features)
df.to_csv(OUTPUT_PATH, index=False)

print("Saved features:", df.shape)