import cv2
import numpy as np

class MotionDetector:
    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.prev_gray = None
        self.mask = None
        self.frame_count = 0

    def detect_motion(self, frame, threshold=500, min_area=100):
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply background subtraction
        fg_mask = self.fgbg.apply(frame)
        
        # Apply some morphology to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Calculate optical flow if we have a previous frame
        flow = None
        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        self.prev_gray = gray
        
        # Create a mask of moving pixels
        if flow is not None:
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            self.mask = magnitude > 1  # Adjust this threshold as needed
        else:
            self.mask = np.zeros(gray.shape, dtype=bool)
        
        # Combine background subtraction and optical flow
        motion_mask = ((fg_mask == 255) | self.mask).astype(np.uint8) * 255
        
        # Find contours of moving areas
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on area
        significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # Calculate motion score
        motion_score = np.sum(motion_mask)
        
        # Determine if there is significant motion based on threshold
        motion_detected = motion_score > threshold or len(significant_contours) > 0
        
        # Only consider motion after a few frames to allow background subtractor to stabilize
        if self.frame_count < 10:
            motion_detected = False
        
        return motion_detected, motion_score, len(significant_contours)

# Global instance of MotionDetector
motion_detector = MotionDetector()

def detect_motion(frame, threshold=500, min_area=100):
    return motion_detector.detect_motion(frame, threshold, min_area)