# AudioFlux-Intelligence

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
├── venv/ # Python virtual environment
├── requirements.txt # Python dependencies
├── README.md # This file
└── .gitignore


---

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/<yourusername>/AudioFlux-Intelligence.git
cd AudioFlux-Intelligence

Create virtual environment and install dependencies:

python3 -m venv venv
source venv/bin/activate    # On Windows use: venv\Scripts\activate
pip install --upgrade pip
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
Trains models on PCA-reduced features and saves model checkpoints in src/models/.

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

Preprocessing: scaling, PCA (50 components), cleaning missing values

Results
Overall Metrics
Model	Accuracy	Precision (macro)	Recall (macro)	F1-score (macro)
Logistic Regression	0.59	0.44	0.33	0.36
Random Forest	0.61	0.67	0.32	0.35
Gradient Boosting	0.59	0.41	0.33	0.35
Per-Genre F1-Scores
Genre	Logistic Regression	Random Forest	Gradient Boosting
Blues	0.00	0.00	0.00
Classical	0.67	0.75	0.67
Country	0.07	0.05	0.07
Easy Listening	0.00	0.00	0.00
Electronic	0.61	0.62	0.61
Experimental	0.58	0.60	0.58
Folk	0.49	0.53	0.49
Hip-Hop	0.55	0.54	0.55
Instrumental	0.22	0.17	0.22
International	0.35	0.27	0.35
Jazz	0.11	0.10	0.11
Old-Time / Historic	0.82	0.87	0.82
Pop	0.07	0.03	0.07
Rock	0.72	0.73	0.72
Soul-RnB	0.03	0.06	0.03
Spoken	0.30	0.31	0.30

Notes:

Random Forest slightly outperforms Gradient Boosting in overall accuracy.

Performance varies significantly across genres due to class imbalance.

PCA helps reduce dimensionality and improve model generalization.
