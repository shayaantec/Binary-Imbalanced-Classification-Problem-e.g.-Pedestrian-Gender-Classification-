import cv2
import os

def preprocess_image(image_path, size=(224, 224)):
    img = cv2.imread(image_path)

    if img is None:
        return None

    # Resize
    img = cv2.resize(img, size)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Histogram Equalization
    equalized = cv2.equalizeHist(gray)

    return equalized

def preprocess_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            processed = preprocess_image(input_path)

            if processed is not None:
                cv2.imwrite(output_path, processed)

if __name__ == "__main__":
    input_male = r"D:\Desktop\dip_proj\MIT-IB\male1"
    input_female = r"D:\Desktop\dip_proj\MIT-IB\female_augmented"
    output_male = r"D:\Desktop\dip_proj\MIT-IB\preprocessed\male1"
    output_female = r"D:\Desktop\dip_proj\MIT-IB\preprocessed\female"

    preprocess_images(input_male, output_male)
    preprocess_images(input_female, output_female)
