import os
import cv2
import glob

# Directory path containing video category folders
videos_dir = r'c:\Users\roksh\OneDrive\Desktop\AI LAB\SignLanguage\BigDataset'

# Output directory to save frames
frames_output_dir = r'C:\Users\roksh\OneDrive\Desktop\AI LAB\SignLanguage\AddingExtracted'
os.makedirs(frames_output_dir, exist_ok=True)

# Create a mapping of category names to labels
category_to_label = {category_name: label for label, category_name in enumerate(os.listdir(videos_dir))}

# Iterate through video files and extract frames
for category_name in os.listdir(videos_dir):
    category_dir = os.path.join(videos_dir, category_name)
    label = category_to_label[category_name]
    
    video_files = glob.glob(os.path.join(category_dir, '*.mp4'))
    for video_file in video_files:
        cap = cv2.VideoCapture(video_file)
        frame_count = 0
        
        # Create a subdirectory for the video's frames
        video_frames_dir = os.path.join(frames_output_dir, category_name, os.path.basename(video_file)[:-4])
        os.makedirs(video_frames_dir, exist_ok=True)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_path = os.path.join(video_frames_dir, f'frame_{frame_count:04d}.jpg')
            cv2.imwrite(frame_path, frame)
            
            frame_count += 1
        
        cap.release()

print("Frame extraction and labeling completed.")
