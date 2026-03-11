README.md for AudioFlux-Intelligence
# AudioFlux-Intelligence 🎵🧠

AudioFlux-Intelligence is an end-to-end machine learning project for **music genre classification** using audio signal features. It extracts features from audio files, processes them (scaling, PCA), trains multiple ML models, and evaluates them with metrics and visualizations.

---

## Project Structure


AudioFlux-Intelligence/
│
├── data/
│ ├── raw/ # Original datasets (FMA small / tracks.csv / features.csv)
│ ├── processed/ # Cleaned and merged dataset (final_dataset.csv)
│
├── src/
│ ├── datafiles/ # Scripts to preprocess and build final dataset
│ │ └── build_dataset.py
│ ├── models/ # Model training and evaluation scripts
│ │ ├── train_model.py
│ │ └── feature_importance.py
│ └── utils/ # Helper functions for data loading, preprocessing
│
├── notebooks/ # Jupyter notebooks (optional)
│
├── audioenv/ # Conda virtual environment
│
├── requirements.txt # Python dependencies
├── README.md # This file
└── .gitignore


---

## Installation

1. **Clone the repo:**

```bash
git clone https://github.com/<yourusername>/AudioFlux-Intelligence.git
cd AudioFlux-Intelligence

Create virtual environment and install dependencies:

conda create -n audioenv python=3.10
conda activate audioenv
pip install -r requirements.txt
Usage
1. Build Dataset
python src/datafiles/build_dataset.py

This will:

Load audio features (features.csv) and metadata (tracks.csv)

Merge them into a final dataset

Save final_dataset.csv in data/processed/

2. Train Models
python src/models/train_model.py

Supports Logistic Regression, Random Forest, and Gradient Boosting.

Trains models on PCA-reduced features.

Saves model checkpoints in src/models/.

3. Feature Importance
python src/models/feature_importance.py

Plots top PCA features contributing to classification.

Saves feature_importance.png.

4. Evaluation

Classification report and accuracy metrics are printed after training.

Optionally, confusion matrices can be generated for error analysis.

Data

Dataset: Free Music Archive (FMA) Small subset

Audio features: MFCCs, chroma, spectral contrast (~518 features)

Samples: 49,598 audio tracks after preprocessing

Split: Train/Val/Test (80/10/10%)

Preprocessing includes scaling, PCA (50 components), and cleaning missing values.

Results
Model	Accuracy
Logistic Regression	0.59
Random Forest	0.61
Gradient Boosting	0.59