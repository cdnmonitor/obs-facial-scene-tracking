import cv2
import asyncio
import threading
from queue import Queue
from object_detection import detect_objects
from motion_detection import detect_motion
from collections import deque

# Buffers to smooth out detections
detection_buffer = {
    'desk': {'person': deque(maxlen=3), 'motion': deque(maxlen=10)},
    'cnc': {'motion': deque(maxlen=10)}
}
last_scene_change_time = 0
SCENE_CHANGE_COOLDOWN = 1  # 1 second cooldown

# Queues for each camera
frame_queues = {}

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

async def process_camera_feeds(obs, config):
    if 'cameras' not in config or not config['cameras']:
        print("No cameras configured. Please run the setup client to add cameras.")
        return

    list_condition_rules(config)

    # Start capture threads
    for name, url in config['cameras'].items():
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
        return sum(detection_buffer[camera][detection_type]) / len(detection_buffer[camera][detection_type]) > 0.7
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
            person_detected = 'person' in detect_objects(frame, confidence_threshold=0.6)
            detection_results[f"{camera}_person"] = person_detected
            print(f"[DEBUG] Person detection for {camera}: {person_detected}")

            motion_threshold = 15000 if camera == 'cnc' else 20000  # Higher threshold for CNC
            motion_detected = detect_motion(frame, threshold=motion_threshold, min_area=1000)
            detection_results[f"{camera}_motion"] = motion_detected
            print(f"[DEBUG] Motion detection for {camera}: {motion_detected}")
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
                print(f"[{timestamp:.3f}] [WARNING] Camera '{camera}' not found in frames. Skipping this condition.")
                all_conditions_met = False
                break

            detection_key = f"{camera}_{detection_type}"
            if detection_key not in detection_results:
                print(f"[{timestamp:.3f}] [WARNING] Detection result for '{detection_key}' not found. Skipping this condition.")
                all_conditions_met = False
                break

            detection_result = detection_results[detection_key]
            smoothed_result = update_detection_buffer(camera, detection_type, detection_result)

            condition_met = (
                (condition_type == 'presence' and smoothed_result) or
                (condition_type == 'absence' and not smoothed_result)
            )

            presence_absence = "is" if condition_type == 'presence' else "is not"
            print(f"[{timestamp:.3f}] [DEBUG] Condition: {camera} {presence_absence} detecting {detection_type}")
            print(f"[{timestamp:.3f}] [DEBUG] Raw Result: {detection_result}, Smoothed Result: {smoothed_result}, Condition Met: {condition_met}")

            if not condition_met:
                all_conditions_met = False
                break

        print(f"[{timestamp:.3f}] [DEBUG] All conditions in set {i} met: {all_conditions_met}")
        print(f"[{timestamp:.3f}] [DEBUG] Scene to switch to if all conditions are met: {condition_set['scene']}")

        if all_conditions_met:
            current_scene = await obs.get_current_scene()
            if current_scene != condition_set['scene']:
                await obs.switch_scene(condition_set['scene'])
                print(f"[{timestamp:.3f}] [DEBUG] Switching to scene '{condition_set['scene']}' based on met conditions")
                last_scene_change_time = current_time
            else:
                print(f"[{timestamp:.3f}] [DEBUG] Already in correct scene '{current_scene}' based on met conditions")
            return  # Exit after first matched condition set

    current_scene = await obs.get_current_scene()
    print(f"[{timestamp:.3f}] [DEBUG] No condition sets fully met. Current scene: {current_scene}")



def list_condition_rules(config):
    print("\nCurrent Condition Rules:")
    for i, condition_set in enumerate(config['logic_conditions'], 1):
        print(f"Rule Set {i}:")
        for j, condition in enumerate(condition_set['conditions'], 1):
            operator = f" AND " if j > 1 else ""
            presence_absence = "is" if condition['condition_type'] == 'presence' else "is not"
            print(f"  {operator}If {condition['camera']} {presence_absence} detecting {condition['detection_type']}")
        print(f"  Then switch to scene: {condition_set['scene']}")
    print()