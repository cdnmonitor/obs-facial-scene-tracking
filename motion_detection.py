import cv2
import numpy as np

prev_frame = None
motion_history = np.zeros((10,))

def detect_motion(frame, threshold=500, min_area=25):
    global prev_frame, motion_history
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    if prev_frame is None:
        prev_frame = gray
        return False
    
    # Ensure prev_frame and gray have the same shape
    if prev_frame.shape != gray.shape:
        prev_frame = cv2.resize(prev_frame, (gray.shape[1], gray.shape[0]))
    
    # Compute the absolute difference between the current frame and the previous frame
    frame_diff = cv2.absdiff(prev_frame, gray)
    
    # Apply a threshold to the difference
    _, thresh = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)
    
    # Calculate the number of changed pixels
    motion_score = np.sum(thresh)
    
    # Update motion history
    motion_history = np.roll(motion_history, 1)
    motion_history[0] = 1 if motion_score > threshold else 0
    
    # Consider motion detected if it's present in any of the recent frames
    smoothed_motion = np.mean(motion_history) > 0.3
    
    # Update the previous frame
    prev_frame = gray
    
    # Debug output
    print(f"[DEBUG] Motion detection - Raw score: {motion_score}, Threshold: {threshold}, "
          f"Smoothed: {smoothed_motion}, Mean history: {np.mean(motion_history):.2f}")
    
    return smoothed_motion