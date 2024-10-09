import asyncio
import simpleobsws
import json
import cv2
import torch
import time
import numpy as np

# Load OBS connection parameters from external files
with open('obs_config.json', 'r') as config_file:
    config = json.load(config_file)
    OBS_URL = config['url']
    OBS_PASSWORD = config['password']
    DESKTOP_CAM_URL = config['desktop_camera_url']
    CNC_CAM_URL = config['cnc_camera_url']

# OBS connection parameters
parameters = simpleobsws.IdentificationParameters()
parameters.rpc_version = 1
ws = simpleobsws.WebSocketClient(url=OBS_URL, password=OBS_PASSWORD, identification_parameters=parameters)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_objects(frame, confidence_threshold=0.5):
    # Perform detection with a confidence threshold
    results = model(frame)
    results = results.pandas().xyxy[0]
    detected_classes = results[results['confidence'] > confidence_threshold]['name'].values.tolist()
    return detected_classes

def detect_motion(prev_frame, current_frame, threshold=15000):
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    # Compute absolute difference between frames
    diff = cv2.absdiff(prev_gray, current_gray)
    # Threshold the difference
    _, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    # Sum the thresholded difference
    motion_score = np.sum(diff_thresh)
    print(f"[DEBUG] Motion score: {motion_score}")
    return motion_score

async def connect_to_obs():
    try:
        print("[DEBUG] Attempting to connect to OBS WebSocket...")
        await ws.connect()
        await ws.wait_until_identified()
        print("[DEBUG] Connected to OBS WebSocket and identified successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to connect to OBS WebSocket: {e}")

async def list_scenes():
    try:
        # Ensure we are connected and identified
        if not ws.identified:
            print("[ERROR] Not identified with OBS WebSocket. Cannot make requests.")
            return []
        
        request = simpleobsws.Request('GetSceneList')
        response = await ws.call(request)
        if response.ok():
            scenes = response.responseData['scenes'][::-1]  # Reverse the order of scenes to match OBS UI
            print("[DEBUG] Available scenes:")
            for scene in scenes:
                print(f"- {scene['sceneName']}")
            return scenes
        else:
            print(f"[ERROR] Failed to get scene list: {response.responseData}")
            return []
    except Exception as e:
        print(f"[ERROR] Exception occurred while listing scenes: {e}")
        return []

async def switch_scene(scene_name, current_scene):
    if scene_name == current_scene:
        print(f"[DEBUG] Scene already set to: {scene_name}, no switch needed.")
        return
    try:
        request = simpleobsws.Request('SetCurrentProgramScene', {
            "sceneName": scene_name
        })
        response = await ws.call(request)
        if response.ok():
            print(f"[DEBUG] Successfully switched to scene: {scene_name}")
        else:
            print(f"[ERROR] Failed to switch to scene {scene_name}: {response.responseData}")
    except Exception as e:
        print(f"[ERROR] Exception occurred while switching to scene {scene_name}: {e}")

async def get_current_scene():
    try:
        request = simpleobsws.Request('GetCurrentProgramScene')
        response = await ws.call(request)
        if response.ok():
            return response.responseData['currentProgramSceneName']
        else:
            print(f"[ERROR] Failed to get current scene: {response.responseData}")
            return None
    except Exception as e:
        print(f"[ERROR] Exception occurred while getting current scene: {e}")
        return None

async def process_camera_feed():
    scenes = await list_scenes()
    if len(scenes) < 4:
        print("[ERROR] Not enough scenes available to perform the required switches.")
        return

    first_scene = scenes[0]['sceneName']
    desktop_cnc_scene = scenes[1]['sceneName']
    machining_scene = scenes[2]['sceneName']
    machining_scene_2 = scenes[3]['sceneName']

    cnc_in_use = False
    prev_cnc_frame = None
    motion_scores = []
    motion_buffer_size = 5
    consecutive_motion_frames_required = 3
    consecutive_no_motion_frames_required = 3
    consecutive_motion_frames = 0
    consecutive_no_motion_frames = 0

    try:
        while True:
            current_scene = await get_current_scene()

            # Capture frame from desktop camera
            cap = cv2.VideoCapture(DESKTOP_CAM_URL)
            ret, frame = cap.read()
            if ret:
                # Show the frame for debugging purposes
                cv2.imshow('Desktop Camera Feed', frame)
                detected_objects = detect_objects(frame)
                print(f"[DEBUG] Detected objects: {detected_objects}")

                if 'person' in detected_objects:
                    # If person is detected in desktop camera
                    cap_cnc = cv2.VideoCapture(CNC_CAM_URL)
                    ret_cnc, frame_cnc = cap_cnc.read()
                    if ret_cnc:
                        cnc_detected_objects = detect_objects(frame_cnc)
                        if 'hand' in cnc_detected_objects or 'arm' in cnc_detected_objects:
                            # Switch to Machining Setup if hands or arms are detected in CNC camera
                            await switch_scene(machining_scene, current_scene)
                        else:
                            if prev_cnc_frame is not None:
                                motion_score = detect_motion(prev_cnc_frame, frame_cnc)
                                motion_scores.append(motion_score)
                                if len(motion_scores) > motion_buffer_size:
                                    motion_scores.pop(0)
                                avg_motion_score = sum(motion_scores) / len(motion_scores)
                                print(f"[DEBUG] Average motion score: {avg_motion_score}")
                                
                                if avg_motion_score > 20000:
                                    consecutive_motion_frames += 1
                                    consecutive_no_motion_frames = 0
                                else:
                                    consecutive_no_motion_frames += 1
                                    consecutive_motion_frames = 0
                                
                                if consecutive_motion_frames >= consecutive_motion_frames_required:
                                    cnc_in_use = True
                                    if current_scene != desktop_cnc_scene:
                                        await switch_scene(desktop_cnc_scene, current_scene)
                                elif consecutive_no_motion_frames >= consecutive_no_motion_frames_required:
                                    cnc_in_use = False
                                    await switch_scene(first_scene, current_scene)
                            prev_cnc_frame = frame_cnc
                    cap_cnc.release()
                elif current_scene == machining_scene:
                    # Check CNC camera if current scene is Machining Setup
                    cap_cnc = cv2.VideoCapture(CNC_CAM_URL)
                    ret_cnc, frame_cnc = cap_cnc.read()
                    if ret_cnc:
                        cnc_detected_objects = detect_objects(frame_cnc)
                        if 'hand' in cnc_detected_objects or 'arm' in cnc_detected_objects:
                            # Stay on Machining Setup if hands or arms are detected
                            await switch_scene(machining_scene, current_scene)
                        else:
                            if prev_cnc_frame is not None:
                                motion_score = detect_motion(prev_cnc_frame, frame_cnc)
                                motion_scores.append(motion_score)
                                if len(motion_scores) > motion_buffer_size:
                                    motion_scores.pop(0)
                                avg_motion_score = sum(motion_scores) / len(motion_scores)
                                print(f"[DEBUG] Average motion score: {avg_motion_score}")
                                
                                if avg_motion_score > 20000:
                                    consecutive_motion_frames += 1
                                    consecutive_no_motion_frames = 0
                                else:
                                    consecutive_no_motion_frames += 1
                                    consecutive_motion_frames = 0
                                
                                if consecutive_motion_frames >= consecutive_motion_frames_required:
                                    cnc_in_use = True
                                    await switch_scene(machining_scene_2, current_scene)
                                elif consecutive_no_motion_frames >= consecutive_no_motion_frames_required:
                                    cnc_in_use = False
                        prev_cnc_frame = frame_cnc
                    cap_cnc.release()
                elif current_scene == machining_scene_2:
                    # Check CNC camera if current scene is Machining Setup 2
                    cap_cnc = cv2.VideoCapture(CNC_CAM_URL)
                    ret_cnc, frame_cnc = cap_cnc.read()
                    if ret_cnc:
                        cnc_detected_objects = detect_objects(frame_cnc)
                        if 'hand' in cnc_detected_objects or 'arm' in cnc_detected_objects:
                            # Switch to Machining Setup if hands or arms are detected in CNC camera
                            await switch_scene(machining_scene, current_scene)
                        else:
                            if prev_cnc_frame is not None:
                                motion_score = detect_motion(prev_cnc_frame, frame_cnc)
                                motion_scores.append(motion_score)
                                if len(motion_scores) > motion_buffer_size:
                                    motion_scores.pop(0)
                                avg_motion_score = sum(motion_scores) / len(motion_scores)
                                print(f"[DEBUG] Average motion score: {avg_motion_score}")
                                
                                if avg_motion_score <= 10000:
                                    consecutive_no_motion_frames += 1
                                    consecutive_motion_frames = 0
                                else:
                                    consecutive_motion_frames += 1
                                    consecutive_no_motion_frames = 0
                                
                                if consecutive_no_motion_frames >= consecutive_no_motion_frames_required:
                                    cnc_in_use = False
                                    await switch_scene(machining_scene, current_scene)
                        prev_cnc_frame = frame_cnc
                    cap_cnc.release()
                else:
                    # Default to Machining Setup if no person in desktop camera and CNC is not in use
                    await switch_scene(machining_scene, current_scene)

            cap.release()

            # Check every 1 second
            await asyncio.sleep(1)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("[DEBUG] Stopping camera processing.")
    finally:
        cv2.destroyAllWindows()

async def main():
    # Connect to OBS WebSocket
    await connect_to_obs()

    # Start processing camera feeds
    await process_camera_feed()

    # Disconnect from OBS WebSocket
    await ws.disconnect()
    print("[DEBUG] Disconnected from OBS WebSocket.")

# Run the main event loop
asyncio.run(main())