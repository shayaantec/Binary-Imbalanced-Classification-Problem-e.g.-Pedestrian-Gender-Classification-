import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load features and assign labels explicitly
def load_features(feature_folder, label):
    features = []
    labels = []
    for filename in os.listdir(feature_folder):
        if filename.lower().endswith('.npy'):
            feature_path = os.path.join(feature_folder, filename)
            feature = np.load(feature_path)
            features.append(feature)
            labels.append(label)
    return np.array(features), np.array(labels)

def feature_fusion_and_pca(male_feature_folder, female_feature_folder):
    # Load male and female features
    male_features, male_labels = load_features(male_feature_folder, label=0)
    female_features, female_labels = load_features(female_feature_folder, label=1)

    # Combine features and labels
    all_features = np.concatenate([male_features, female_features], axis=0)
    all_labels = np.concatenate([male_labels, female_labels], axis=0)

    # Standardize
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)

    # Apply PCA
    pca = PCA(n_components=50)  # Adjust as needed
    reduced_features = pca.fit_transform(all_features_scaled)

    # Save
    np.save("reduced_features.npy", reduced_features)
    np.save("labels.npy", all_labels)
    print("Features have been reduced and saved!")

if __name__ == "__main__":
    male_feature_folder = r"D:\Desktop\dip_proj\MIT-IB\features\male_vgg19"
    female_feature_folder = r"D:\Desktop\dip_proj\MIT-IB\features\female_vgg19"

    feature_fusion_and_pca(male_feature_folder, female_feature_folder)
