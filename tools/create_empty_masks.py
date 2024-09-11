import numpy as np
import cv2
import os

def create_empty_masks(frame_folder, mask_folder):
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)
    
    frame_files = [f for f in os.listdir(frame_folder) if f.endswith('.jpg')]
    
    for frame_file in frame_files:
        frame_path = os.path.join(frame_folder, frame_file)
        img = cv2.imread(frame_path)
        height, width, _ = img.shape
        
        mask = np.zeros((height, width), dtype=np.uint8)
        mask_path = os.path.join(mask_folder, frame_file)
        cv2.imwrite(mask_path, mask)

frame_folder = "C:/Users/BT/segment-anything-2/videos/input_frames"
mask_folder = "C:/Users/BT/segment-anything-2/videos/input_masks"
create_empty_masks(frame_folder, mask_folder)
