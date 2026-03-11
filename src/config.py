import os
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

DATA_PATH = "data/processed/final_dataset.csv"
RESULTS_PATH = "results/experiment_results.csv"
MODEL_PATH = "models/"