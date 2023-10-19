import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import os
import time

# Initialize the HandDetector
detector = HandDetector(maxHands=1)

# Offset and image size parameters
offset = 20
imgSize = 300

# Specify the root directory where your images are stored
root_dir = "/content/drive/MyDrive/Preprocessed_Frames" # CHANGE THIS

# Create an output folder for the cropped hand images
output_root = "/content/drive/MyDrive/Preprocessed_Frames_Hands" #CHANGE THIS
os.makedirs(output_root, exist_ok=True)

# Process each category (e.g., 'eat', 'milk')
categories = os.listdir(root_dir)

for category in categories:
    category_path = os.path.join(root_dir, category)

    # Process each video within the category
    videos = os.listdir(category_path)

    for video in videos:
        video_path = os.path.join(category_path, video)

        # Create an output folder for the current video
        output_category_folder = os.path.join(output_root, category)
        output_video_folder = os.path.join(output_category_folder, video)
        os.makedirs(output_video_folder, exist_ok=True)

        # Process each frame in the video
        frames = os.listdir(video_path)

        for frame_filename in frames:
            frame_path = os.path.join(video_path, frame_filename)

            # Load the image
            img = cv2.imread(frame_path)

            # Find hands in the image
            hands, img = detector.findHands(img)

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                # Resize and place the cropped hand image into a white canvas
                imgCropShape = imgCrop.shape
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                # Save the cropped hand image
                output_filename = os.path.splitext(frame_filename)[0] + "_hand.jpg"
                output_path = os.path.join(output_video_folder, output_filename)
                cv2.imwrite(output_path, imgWhite)

                print(f"Saved cropped hand image: {output_path}")

print("All frames processed and cropped.")