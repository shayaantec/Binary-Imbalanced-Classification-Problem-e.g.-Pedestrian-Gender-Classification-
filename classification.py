import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

# Load the reduced features and labels
features = np.load("reduced_features.npy")
labels = np.load("labels.npy")

# Print class distribution
unique_classes, counts = np.unique(labels, return_counts=True)
print("Class distribution:", dict(zip(unique_classes, counts)))

# Ensure both classes are present
if len(unique_classes) < 2:
    raise ValueError("Both male and female classes must be present for classification.")

# Define Linear SVM classifier
clf = SVC(kernel='linear', C=1)

# K-Fold Cross-Validation (k=10)
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Predict using cross-validation
y_pred = cross_val_predict(clf, features, labels, cv=kfold)

# Compute metrics
accuracy = accuracy_score(labels, y_pred) * 100
precision = precision_score(labels, y_pred) * 100
recall = recall_score(labels, y_pred) * 100
f1 = f1_score(labels, y_pred) * 100

# Print classification report
print("\nClassification Report:\n", classification_report(labels, y_pred, target_names=["Male", "Female"]))
print("Accuracy :", f"{accuracy:.2f}%")
print("Precision:", f"{precision:.2f}%")
print("Recall   :", f"{recall:.2f}%")
print("F1 Score :", f"{f1:.2f}%")

# Confusion matrix
cm = confusion_matrix(labels, y_pred)

# Plot and save confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Male", "Female"], yticklabels=["Male", "Female"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

# Save the confusion matrix image
plt.savefig("confusion_matrix.png")

# Show the confusion matrix plot
plt.show()
