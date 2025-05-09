import cv2
import numpy as np
import os

def augment_image(image):
    """
    Returns: [original, flipped, rotated]
    """
    flipped = cv2.flip(image, 1)
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return [image, flipped, rotated]

def balance_female_class(input_folder, output_folder, target_count=600):
    """
    Augments only the female class to match the number of male samples (600).
    """
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    current_index = 0
    for i, filename in enumerate(image_files):
        if current_index >= target_count:
            break

        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        augmented_imgs = augment_image(img)

        for j, aug in enumerate(augmented_imgs):
            if current_index >= target_count:
                break
            save_path = os.path.join(output_folder, f"female_{current_index}.jpg")
            cv2.imwrite(save_path, aug)
            current_index += 1

    print(f"Generated {current_index} balanced female samples.")

if __name__ == "__main__":
    female_input = r"D:\Desktop\dip_proj\MIT-IB\female"
    female_output = r"D:\Desktop\dip_proj\MIT-IB\female_augmented"

    balance_female_class(female_input, female_output)
