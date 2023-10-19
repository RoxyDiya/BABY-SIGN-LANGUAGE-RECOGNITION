import os
import cv2
import random
import numpy as np

# Set up paths
input_directory = r"C:\Users\roksh\OneDrive\Desktop\AI LAB\SignLanguage\FramesExtracted"  # Replace with your dataset path
output_directory = r"C:\Users\roksh\OneDrive\Desktop\AI LAB\SignLanguage\Augmented3"  # Replace with desired output path

# List of non-disruptive augmentation techniques
def apply_color_jitter(img):
    # Apply slight color jitter
    jitter_amount = np.random.randint(-10, 10)
    img = img.astype(np.int16)
    img += jitter_amount
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def apply_gaussian_noise(img):
    # Add Gaussian noise
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    img = img.astype(np.int16) + noise
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def apply_horizontal_flip(img):
    return cv2.flip(img, 1)

def apply_vertical_flip(img):
    return cv2.flip(img, 0)

def apply_background_removal(img):
    fgbg = cv2.createBackgroundSubtractorMOG2()
    fgmask = fgbg.apply(img)
    masked_frame = cv2.bitwise_and(img, img, mask=fgmask)
    return masked_frame

def apply_random_rotation(img):
    angle = random.randint(-60, 60)  # Rotate by random angle between -30 and 30 degrees
    rows, cols, _ = img.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    return cv2.warpAffine(img, rotation_matrix, (cols, rows))

def apply_brightness_contrast(img):
    alpha = random.uniform(0.8, 1.2)  # Brightness adjustment
    beta = random.randint(-20, 20)    # Contrast adjustment
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

augmentation_functions = [
    lambda img: apply_color_jitter(img),
    lambda img: apply_gaussian_noise(img)
    #lambda img: apply_background_removal(img)
    #Add more augmentation functions here
]

# Iterate through classes and videos
# Iterate through classes and videos
for class_name in os.listdir(input_directory):
    class_path = os.path.join(input_directory, class_name)
    output_class_path = os.path.join(output_directory, class_name)
    os.makedirs(output_class_path, exist_ok=True)
    
    for video_folder in os.listdir(class_path):
        video_folder_path = os.path.join(class_path, video_folder)
        output_video_folder_path = os.path.join(output_class_path, video_folder)
        os.makedirs(output_video_folder_path, exist_ok=True)
        
        for frame_name in os.listdir(video_folder_path):
            frame_path = os.path.join(video_folder_path, frame_name)
            frame = cv2.imread(frame_path)
            
            # Apply random augmentation function to the frame
            augmentation_func = random.choice(augmentation_functions)
            augmented_frame = augmentation_func(frame)
            
            # Save augmented frame
            output_frame_path = os.path.join(output_video_folder_path, f"{frame_name.split('.')[0]}_augmented.jpg")
            cv2.imwrite(output_frame_path, augmented_frame)

print("Data augmentation complete.")
