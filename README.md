# AudioFlux Intelligence

AudioFlux Intelligence is a machine learning pipeline for **music genre classification and audio feature analysis** using the Free Music Archive (FMA) dataset.

The project builds a complete ML workflow including:

* dataset construction
* preprocessing and feature engineering
* dimensionality reduction
* training multiple models
* evaluation and error analysis

The system compares several models to understand how well **signal-processing-based audio features** can predict music genres.



# Project Architecture

```
Raw FMA Dataset
      │
      ▼
Feature Dataset (features.csv + tracks.csv)
      │
      ▼
Dataset Builder
(src/datafiles/build_dataset.py)
      │
      ▼
Preprocessing Pipeline
(src/datafiles/preprocess_dataset.py)

• NaN handling  
• Feature scaling (StandardScaler)  
• Dimensionality reduction (PCA)

      │
      ▼
Model Training
(src/models)

• Logistic Regression
• Random Forest
• Gradient Boosting

      │
      ▼
Evaluation + Metrics
```



# Dataset

Dataset used: **Free Music Archive (FMA)**

Sources:

* features.csv → extracted audio signal features
* tracks.csv → track metadata and genre labels

Dataset statistics after processing:

| Property     | Value               |
| ------------ | ------------------- |
| Total tracks | 49,598              |
| Features     | ~518 audio features |
| Genres       | 16                  |
| Test samples | 9,920               |

Audio features include:

* MFCC coefficients
* spectral centroid
* spectral bandwidth
* chroma features
* spectral contrast
* tonnetz harmonic features

These features capture the **spectral and harmonic structure of music**.



# Preprocessing Pipeline

The following preprocessing steps were applied:

1. Merge audio features with genre labels
2. Remove tracks with missing genre labels
3. Handle missing values
4. Standardize features using `StandardScaler`
5. Apply **PCA dimensionality reduction**

Final dataset:

| Component        | Value   |
| ---------------- | ------- |
| PCA features     | 50      |
| Training samples | ~39,600 |
| Test samples     | 9,920   |

PCA was used to reduce noise and improve model training efficiency.



# Models Trained

Three models were trained and compared:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

Each model was trained using an **80/20 stratified train-test split**.

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-score



# Model Performance

| Model               | Accuracy | Macro F1 | Weighted F1 |
| ------------------- | -------- | -------- | ----------- |
| Logistic Regression | 0.59     | 0.35     | 0.56        |
| Random Forest       | **0.61** | 0.35     | **0.57**    |
| Gradient Boosting   | 0.59     | 0.35     | 0.56        |

Best performing model: **Random Forest**


# Key Observations

### 1. Class imbalance

Some genres have many samples:

* Rock
* Electronic
* Experimental

While others have very few:

* Easy Listening
* Blues
* Soul-RnB

This causes models to prioritize large genres and ignore smaller ones.


### 2. Strong genre separation

Certain genres have distinctive acoustic signatures:

Examples:

* Classical
* Old-Time / Historic
* Rock

These achieved relatively high F1 scores.



### 3. Confusion between related genres

Genres with similar acoustic characteristics were often confused:

Examples:

* Pop vs Electronic
* Blues vs Rock
* Country vs Folk

This reflects overlapping audio feature distributions.


# Example Classification Results

High performing genres:

| Genre               | F1 Score |
| ------------------- | -------- |
| Rock                | ~0.73    |
| Classical           | ~0.67    |
| Old-Time / Historic | ~0.82    |

Low performing genres:

| Genre          | F1 Score |
| -------------- | -------- |
| Blues          | 0.00     |
| Easy Listening | 0.00     |
| Soul-RnB       | ~0.03    |

These results highlight the impact of **dataset imbalance**.



# Repository Structure

```
AudioFlux-Intelligence
 |
data
 ├─ raw
 │   ├─ features.csv
 │   └─ tracks.csv
 │
 └─ processed
 |   ├─ final_dataset.csv
 |   └─ model_dataset.csv
 |
src
 ├─ datafiles
 │   ├─ build_dataset.py
 │   └─ preprocess_dataset.py
 │
 └─ models
 |   ├─ train_logistic.py
 |   ├─ train_randomforest.py
 |   └─ train_gboost.py
 |
models
 └─ saved trained models
 |
README.md
requirements.txt
```



# Running the Project

### 1. Build dataset

```
python src/datafiles/build_dataset.py
```

### 2. Preprocess dataset

```
python src/datafiles/preprocess_dataset.py
```

### 3. Train models

```
python src/models/train_logistic.py
python src/models/train_randomforest.py
python src/models/train_gboost.py
```


# Future Improvements

Possible improvements include:

* Class imbalance handling (SMOTE / class weights)
* Audio based deep learning models for audio spectrograms
* ASR with LLM for the feature and lyrical analysis

