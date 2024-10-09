import cv2
import mediapipe as mp
import asyncio
from object_detection import detect_objects
from motion_detection import detect_motion

# Initialize Mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def detect_hands_with_mediapipe(frame):
    # Convert the image to RGB as Mediapipe requires this format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    hand_detected = False
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_detected = True  # Set flag to true if hand landmarks are detected

    return hand_detected

async def process_camera_feed(obs, desktop_cam_url, cnc_cam_url):
    scenes = await obs.list_scenes()
    # Ensure scenes are in reverse order
    scenes = scenes[::-1]

    if len(scenes) < 4:
        print("[ERROR] Not enough scenes available to perform the required switches.")
        return

    first_scene = scenes[0]['sceneName']  # Machining Setup 2
    machining_scene = scenes[1]['sceneName']  # Machining Setup
    desktop_cnc_scene = scenes[2]['sceneName']  # Desktop Setup + CNC
    desktop_scene = scenes[3]['sceneName']  # Desktop Setup

    prev_cnc_frame = None
    motion_scores = []
    motion_buffer_size = 5

    try:
        while True:
            current_scene = await obs.get_current_scene()
            print(f"[DEBUG] Current scene during loop: {current_scene}")

            # Capture frame from desktop camera
            cap = cv2.VideoCapture(desktop_cam_url)
            ret, frame = cap.read()
            if ret:
                cv2.imshow('Desktop Camera Feed', frame)
                detected_objects = detect_objects(frame)
                print(f"[DEBUG] Detected objects in desktop camera: {detected_objects}")

                # PRIORITY: If a person is detected in the desktop feed, lock to the Desktop Setup scenes
                if 'person' in detected_objects:
                    print("[DEBUG] Person detected in desktop camera.")
                    await obs.switch_scene(desktop_scene)
                    await handle_cnc_feed(obs, cnc_cam_url, current_scene, prev_cnc_frame, motion_scores,
                                          desktop_scene, desktop_cnc_scene, first_scene, machining_scene, motion_buffer_size)
                else:
                    print("[DEBUG] No person detected in desktop camera.")
                    if await check_cnc_for_hands_or_motion(obs, cnc_cam_url, current_scene, machining_scene, first_scene):
                        pass  # Already handled inside the function
                    else:
                        print("[DEBUG] No person detected and no significant activity in CNC. Switching to Machining Setup 2.")
                        await obs.switch_scene(first_scene)

            cap.release()
            await asyncio.sleep(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[DEBUG] User requested to stop camera processing.")
                break
    finally:
        cv2.destroyAllWindows()

async def handle_cnc_feed(obs, cnc_cam_url, current_scene, prev_cnc_frame, motion_scores, desktop_scene, 
                          desktop_cnc_scene, first_scene, machining_scene, motion_buffer_size):
    cap_cnc = cv2.VideoCapture(cnc_cam_url)
    ret_cnc, frame_cnc = cap_cnc.read()
    if ret_cnc:
        hand_detected = detect_hands_with_mediapipe(frame_cnc)
        print(f"[DEBUG] Hand detected in CNC camera: {hand_detected}")

        if hand_detected:
            print(f"[DEBUG] Hand detected in CNC camera. Switching to Machining Setup scene.")
            await obs.switch_scene(machining_scene)
        else:
            motion_score = detect_motion(prev_cnc_frame, frame_cnc) if prev_cnc_frame is not None else 0
            motion_scores.append(motion_score)
            if len(motion_scores) > motion_buffer_size:
                motion_scores.pop(0)
            avg_motion_score = sum(motion_scores) / len(motion_scores)
            print(f"[DEBUG] Average motion score in CNC feed: {avg_motion_score}")

            if avg_motion_score > 20000:
                print(f"[DEBUG] Significant motion detected in CNC. Switching to Desktop Setup + CNC scene.")
                await obs.switch_scene(desktop_cnc_scene)

        prev_cnc_frame = frame_cnc
    else:
        print("[ERROR] Failed to capture frame from CNC camera.")
    cap_cnc.release()


async def check_cnc_for_hands_or_motion(obs, cnc_cam_url, current_scene, machining_scene, first_scene):
    cap_cnc = cv2.VideoCapture(cnc_cam_url)
    ret_cnc, frame_cnc = cap_cnc.read()
    if ret_cnc:
        cnc_detected_objects = detect_objects(frame_cnc)
        print(f"[DEBUG] Detected objects in CNC camera for checking: {cnc_detected_objects}")

        if 'hand' in cnc_detected_objects or 'arm' in cnc_detected_objects:
            print(f"[DEBUG] Hand or arm detected in CNC camera. Switching to Machining Setup.")
            await obs.switch_scene(machining_scene)
            return True
        else:
            print("[DEBUG] No hand or arm detected in CNC camera.")
    else:
        print("[ERROR] Failed to capture frame from CNC camera during hand/motion check.")

    cap_cnc.release()
    return False
