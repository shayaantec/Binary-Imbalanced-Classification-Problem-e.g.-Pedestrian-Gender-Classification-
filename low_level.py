import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from skimage import measure
import os

def extract_hog_features(image):
    # HOG feature extraction
    features, hog_image = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

def extract_lbp_features(image, radius=1, n_points=8):
    # LBP feature extraction
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    return lbp.flatten()

def extract_glcm_features(image):
    # GLCM feature extraction
    glcm = measure.shannon_entropy(image)
    return glcm

def extract_features(input_folder, output_folder, feature_type="HOG"):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue

            if feature_type == "HOG":
                features = extract_hog_features(img)
            elif feature_type == "LBP":
                features = extract_lbp_features(img)
            elif feature_type == "GLCM":
                features = extract_glcm_features(img)

            # Save the features (e.g., as a .txt file or numpy array)
            output_path = os.path.join(output_folder, filename.replace('.jpg', '_features.npy'))
            np.save(output_path, features)

if __name__ == "__main__":
    input_male = r"D:\Desktop\dip_proj\MIT-IB\preprocessed\male1"
    input_female = r"D:\Desktop\dip_proj\MIT-IB\preprocessed\female"
    output_male = r"D:\Desktop\dip_proj\MIT-IB\features\male1"
    output_female = r"D:\Desktop\dip_proj\MIT-IB\features\female"

    # Example for extracting HOG features for both datasets
    extract_features(input_male, output_male, feature_type="HOG")
    extract_features(input_female, output_female, feature_type="HOG")
