# 🧠 Binary Imbalanced Classification: Pedestrian Gender Classification

This project addresses the challenge of **binary classification on an imbalanced dataset**, specifically for classifying the **gender of pedestrians** based on visual features. Various techniques are used including data augmentation, low- and high-level feature extraction, dimensionality reduction, and multiple classification models.

---

## 📁 Project Structure

| File Name              | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `augmented.py`         | Applies data augmentation techniques to balance the dataset                 |
| `classification.py`    | Performs model training, testing, and evaluation using ML classifiers       |
| `confusion_matrix.png` | Visual output showing model performance                                     |
| `data_preprocessing.py`| Handles preprocessing like normalization and image resizing                 |
| `high_level.py`        | Extracts high-level features using pretrained deep learning models          |
| `low_level.py`         | Extracts low-level features such as HOG and color histograms                |
| `pca.py`               | Applies Principal Component Analysis (PCA) for dimensionality reduction     |
| `labels.npy`           | Contains labels (genders) for each pedestrian image                         |
| `reduced_features.npy` | Stores PCA-reduced features to be used in training                         |

---

## ⚙️ Installation & Requirements

Make sure you have Python installed. Then, install the required packages:

```bash
pip install numpy scikit-learn opencv-python matplotlib
