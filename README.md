# 🧠 Binary Imbalanced Classification: Pedestrian Gender Classification

This project tackles the challenge of classifying pedestrian gender from images under an imbalanced dataset scenario. It implements various preprocessing, feature extraction, and machine learning techniques to improve performance.

---

## 📁 Project Structure

- `augmented.py` – Handles data augmentation to balance the dataset.
- `classification.py` – Trains and evaluates classifiers.
- `confusion_matrix.png` – Visual representation of model performance.
- `data_preprocessing.py` – Normalization and resizing of images.
- `high_level.py` – Extraction of high-level (deep) features.
- `labels.npy` – Numpy array storing image labels.
- `low_level.py` – Extraction of handcrafted low-level features.
- `pca.py` – Dimensionality reduction using PCA.
- `reduced_features.npy` – Features after PCA transformation.

---

## ⚙️ Technologies Used

- Python
- NumPy
- OpenCV
- Scikit-learn
- Matplotlib
- Pandas

---

## 📊 Model Evaluation

Performance is measured using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

Imbalanced classification techniques such as **data augmentation** and **class weighting** were applied.

---

## ✅ Features Implemented

- **Image Preprocessing** (Normalization, Resizing)
- **Feature Extraction** (Low-Level and Deep Features)
- **Dimensionality Reduction** (PCA)
- **Model Training & Evaluation** (SVM, KNN, etc.)
- **Confusion Matrix Visualization**
- **Imbalance Handling**

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Binary-Imbalanced-Classification-Problem
   cd Binary-Imbalanced-Classification-Problem
