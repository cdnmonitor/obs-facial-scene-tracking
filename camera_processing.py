import cv2
import asyncio
import threading
from queue import Queue
from object_detection import detect_objects
from motion_detection import detect_motion
from collections import deque

# Buffers to smooth out detections
detection_buffer = {}
last_scene_change_time = 0
SCENE_CHANGE_COOLDOWN = 1  # 1 second cooldown

# Queues for each camera
frame_queues = {}

# Motion detection thresholds
MOTION_THRESHOLD = 10000  # Adjust this value based on testing
MIN_CONTOUR_AREA = 100  # Adjust this value based on testing

def capture_frames(camera_name, url, queue):
    cap = cv2.VideoCapture(url)
    while True:
        ret, frame = cap.read()
        if ret:
            if not queue.empty():
                try:
                    queue.get_nowait()   # Discard previous frame
                except Queue.Empty:
                    pass
            queue.put(frame)
        else:
            print(f"[WARNING] Failed to capture frame from camera: {camera_name}")
        cv2.waitKey(10)  # Small delay to reduce CPU usage

def apply_detection_boundaries(frame, boundaries):
    height, width = frame.shape[:2]
    left = int(boundaries['left'] * width / 100)
    top = int(boundaries['top'] * height / 100)
    right = int(boundaries['right'] * width / 100)
    bottom = int(boundaries['bottom'] * height / 100)
    return frame[top:bottom, left:right]

async def process_camera_feeds(obs, config):
    global detection_buffer
    
    if 'cameras' not in config or not config['cameras']:
        print("No cameras configured. Please run the setup client to add cameras.")
        return

    # Initialize detection buffers based on the logic conditions
    for condition_set in config['logic_conditions']:
        for condition in condition_set['conditions']:
            camera = condition['camera']
            detection_type = condition['detection_type']
            if camera not in detection_buffer:
                detection_buffer[camera] = {}
            if detection_type not in detection_buffer[camera]:
                detection_buffer[camera][detection_type] = deque(maxlen=10)

    # Start capture threads
    for name, camera_info in config['cameras'].items():
        url = camera_info['url'] if isinstance(camera_info, dict) else camera_info
        frame_queues[name] = Queue(maxsize=1)
        threading.Thread(target=capture_frames, args=(name, url, frame_queues[name]), daemon=True).start()

    try:
        while True:
            frames = {}
            for name, queue in frame_queues.items():
                if not queue.empty():
                    frames[name] = queue.get()

            if frames:
                await evaluate_conditions(obs, config, frames)

            await asyncio.sleep(0.1)  # Check every 100ms
    except asyncio.CancelledError:
        print("[DEBUG] Camera processing was cancelled.")
    finally:
        print("[DEBUG] Stopping camera processing.")

def update_detection_buffer(camera, detection_type, result):
    detection_buffer[camera][detection_type].append(result)
    if detection_type == 'motion':
        # For motion, require more consistent detection
        return sum(detection_buffer[camera][detection_type]) / len(detection_buffer[camera][detection_type]) > 0.5
    else:
        # For person detection, keep it more responsive
        return sum(detection_buffer[camera][detection_type]) / len(detection_buffer[camera][detection_type]) > 0.5

async def evaluate_conditions(obs, config, frames):
    global last_scene_change_time
    current_time = asyncio.get_event_loop().time()

    if current_time - last_scene_change_time < SCENE_CHANGE_COOLDOWN:
        return  # Skip evaluation if we're still in the cooldown period

    if 'logic_conditions' not in config or not config['logic_conditions']:
        print("No logic conditions configured. Please run the setup client to add conditions.")
        return

    timestamp = asyncio.get_event_loop().time()

    # Perform detections once for each camera
    detection_results = {}
    for camera, frame in frames.items():
        try:
            camera_info = config['cameras'][camera]
            boundaries = camera_info.get('detection_boundaries') if isinstance(camera_info, dict) else None

            if boundaries:
                frame = apply_detection_boundaries(frame, boundaries)

            # Perform detections based on the conditions for this camera
            for condition_set in config['logic_conditions']:
                for condition in condition_set['conditions']:
                    if condition['camera'] == camera:
                        if condition['detection_type'] == 'person':
                            person_detected = 'person' in detect_objects(frame, confidence_threshold=0.6)
                            detection_results[f"{camera}_person"] = update_detection_buffer(camera, 'person', person_detected)
                            print(f"[DEBUG] Person detection for {camera}: {detection_results[f'{camera}_person']}")
                        elif condition['detection_type'] == 'motion':
                            motion_detected, motion_score, contours_count = detect_motion(frame, threshold=MOTION_THRESHOLD, min_area=MIN_CONTOUR_AREA)
                            detection_results[f"{camera}_motion"] = update_detection_buffer(camera, 'motion', motion_detected)
                            print(f"[DEBUG] Motion detection for {camera} - Motion detected: {motion_detected}, Score: {motion_score}, Contours: {contours_count}")
        except Exception as e:
            print(f"[ERROR] Error in detection for camera {camera}: {str(e)}")
            detection_results[f"{camera}_person"] = False
            detection_results[f"{camera}_motion"] = False

    for i, condition_set in enumerate(config['logic_conditions'], 1):
        print(f"\n[{timestamp:.3f}] [DEBUG] Evaluating Condition Set {i}:")
        all_conditions_met = True

        for condition in condition_set['conditions']:
            camera = condition['camera']
            detection_type = condition['detection_type']
            condition_type = condition['condition_type']

            if camera not in frames:
                all_conditions_met = False
                break

            detection_key = f"{camera}_{detection_type}"
            detection_result = detection_results.get(detection_key, False)

            condition_met = (
                (condition_type == 'presence' and detection_result) or
                (condition_type == 'absence' and not detection_result)
            )

            if not condition_met:
                all_conditions_met = False
                break

        print(f"[{timestamp:.3f}] [DEBUG] All conditions in set {i} met: {all_conditions_met}")
        if all_conditions_met:
            current_scene = await obs.get_current_scene()
            if current_scene != condition_set['scene']:
                await obs.switch_scene(condition_set['scene'])
                print(f"\033[92m[{timestamp:.3f}] [DEBUG] Switching to scene '{condition_set['scene']}' based on met conditions\033[0m")
                last_scene_change_time = current_time
            else:
                print(f"[{timestamp:.3f}] [DEBUG] Already in correct scene '{current_scene}' based on met conditions")
            return  # Exit after switching scene

    current_scene = await obs.get_current_scene()
    print(f"[{timestamp:.3f}] [DEBUG] No condition sets fully met. Current scene: {current_scene}")