# Human-Voice-Classification-and-Clustering

# 🎙️ Human Voice Classification and Clustering

## 🧩 Introduction

This project analyzes human voice characteristics using machine learning to classify gender and group similar voice samples via clustering techniques. We use extracted audio features such as MFCCs, pitch, energy, and spectral properties to uncover patterns in speech data.

## ❗ Problem Statement

Voice classification systems often require complex, resource-intensive audio processing. This project proposes a lightweight ML-based approach that:

- Classifies gender from extracted audio features
- Clusters unlabeled voices based on similarity
- Provides an interactive Streamlit app for ease of use

## 💡 Proposed Solution

We implement a pipeline with the following steps:

### 1. Classification
- Predict gender using top 10 selected audio features
- Trained and evaluated using several models
- Final model: **Support Vector Machine (SVM)**

### 2. Clustering
- Group similar voice samples without labels
- Models used: K-Means, DBSCAN, GMM, Agglomerative, Spectral Clustering
- PCA visualization of cluster results

## 🛠️ Technologies and Languages Used

| Component              | Technology                          |
|------------------------|-------------------------------------|
| Language               | Python                              |
| Data Analysis          | pandas, NumPy                       |
| Visualization          | matplotlib, seaborn                 |
| ML Models              | scikit-learn                        |
| Web App                | Streamlit                           |
| Model Persistence      | pickle                              |

## 🤖 Machine Learning Models

### Classification Models
- Support Vector Machine (SVM) ✅
- K-Nearest Neighbors (KNN)
- Random Forest
- Gradient Boosting
- Neural Networks

### Clustering Models
- K-Means Clustering
- DBSCAN
- Gaussian Mixture Model (GMM)
- Agglomerative Clustering

## 📊 Evaluation Metrics

### Clustering
- Silhouette Score

### Classification
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

## 📈 Project Features

- Preprocessing and scaling of 43 audio features
- Feature selection to reduce to top 10 inputs
- SVM-based prediction via sliders (non-technical UI)
- Leaderboard for model comparison
- Clustering visualizations using PCA
- Fully interactive Streamlit dashboard


