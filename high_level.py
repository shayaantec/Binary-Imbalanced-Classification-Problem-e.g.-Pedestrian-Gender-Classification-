import numpy as np
import os
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model

# Load the VGG19 model and remove the top layers
def load_vgg19_model():
    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)  # fc2 is the FC7 layer
    return model

def extract_vgg19_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))  # VGG19 input size is 224x224
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image for VGG19

    features = model.predict(img_array)
    return features.flatten()  # Flatten the output to 1D array

def extract_deep_features(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    model = load_vgg19_model()

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_folder, filename)
            features = extract_vgg19_features(img_path, model)

            # Save the extracted deep features (e.g., as a .npy file)
            output_path = os.path.join(output_folder, filename.replace('.jpg', '_vgg19_features.npy'))
            np.save(output_path, features)

if __name__ == "__main__":
    input_male = r"D:\Desktop\dip_proj\MIT-IB\preprocessed\male1"
    input_female = r"D:\Desktop\dip_proj\MIT-IB\preprocessed\female"
    output_male = r"D:\Desktop\dip_proj\MIT-IB\features\male_vgg19"
    output_female = r"D:\Desktop\dip_proj\MIT-IB\features\female_vgg19"

    # Extract deep features using VGG19 for both datasets
    extract_deep_features(input_male, output_male)
    extract_deep_features(input_female, output_female)
