import cv2
import numpy as np
from collections import deque

prev_frame = None
motion_history = deque(maxlen=60)  # Increased history length
sustained_motion_counter = 0
SUSTAINED_MOTION_THRESHOLD = 30  # Increased threshold for sustained motion

def detect_motion(frame, threshold=2000000, min_area=5000, blur_size=41, camera_type='default'):
    global prev_frame, motion_history, sustained_motion_counter
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise (increased blur size)
    gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    
    if prev_frame is None:
        prev_frame = gray
        return False
    
    # Ensure prev_frame and gray have the same shape
    if prev_frame.shape != gray.shape:
        prev_frame = cv2.resize(prev_frame, (gray.shape[1], gray.shape[0]))
    
    # Compute the absolute difference between the current frame and the previous frame
    frame_diff = cv2.absdiff(prev_frame, gray)
    
    # Apply a threshold to the difference (increased threshold)
    _, thresh = cv2.threshold(frame_diff, 40, 255, cv2.THRESH_BINARY)
    
    # Dilate the thresholded image to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=3)
    
    # Find contours on thresholded image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate the total area of motion
    motion_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > min_area)
    
    # Calculate the motion score based on the area
    motion_score = motion_area
    
    # Update motion history
    motion_detected = motion_score > threshold
    motion_history.append(motion_detected)
    
    # Check for sustained motion
    if motion_detected:
        sustained_motion_counter += 1
    else:
        sustained_motion_counter = max(0, sustained_motion_counter - 2)  # Decay faster
    
    # Consider motion detected if it's consistently present in recent frames and sustained
    smoothed_motion = (sum(motion_history) / len(motion_history) > 0.8 and 
                       sustained_motion_counter >= SUSTAINED_MOTION_THRESHOLD)
    
    # For CNC camera, require even more sustained motion
    if camera_type == 'cnc':
        smoothed_motion = smoothed_motion and sustained_motion_counter >= SUSTAINED_MOTION_THRESHOLD * 1.5
    
    # Update the previous frame
    prev_frame = gray
    
    # Debug output
    print(f"[DEBUG] Motion detection - Camera: {camera_type}, Raw score: {motion_score}, "
          f"Threshold: {threshold}, Smoothed: {smoothed_motion}, "
          f"Sustained counter: {sustained_motion_counter}, "
          f"Mean history: {sum(motion_history) / len(motion_history):.2f}")
    
    return smoothed_motion