# Automated OBS Scene Switcher

This project provides an automated solution for managing OBS (Open Broadcaster Software) scenes based on camera input. It's designed to assist solo streamers or content creators in managing multiple camera setups without manual intervention.

## Overview

The system uses computer vision techniques, specifically motion detection and person detection, to determine the active area in a multi-camera setup. Based on predefined conditions, it automatically switches OBS scenes, allowing for a dynamic and responsive stream layout. Recommended hardware includes a dedicated streaming computer, and a combination of Raspberry Pi 4/5's with USB cameras or running [MJPG-Streamer](https://github.com/jacksonliam/mjpg-streamer)

## Demo Video

Check out this demo video to see the Automated OBS Scene Switcher in action:

[![Automated OBS Scene Switcher Demo](https://img.youtube.com/vi/ZylC5rzKxhs/0.jpg)](https://www.youtube.com/watch?v=ZylC5rzKxhs)

Click the image above to watch the demo on YouTube.

## Key Features

- Automatic scene switching based on camera activity
- Support for multiple MJPEG streaming cameras
- Person detection for identifying human presence in camera feeds
- Motion detection for identifying activity in camera feeds
- Customizable detection areas within camera feeds
- Flexible logic conditions for scene switching
- Integration with OBS via WebSocket

## Setup

1. Prerequisites:
   - OBS (Open Broadcaster Software) with WebSocket plugin installed
   - Python 3.7 or higher

2. Installation:
   - Clone this repository to your local machine
   - Install required Python packages:
     ```
     pip install opencv-python torch simpleobsws
     ```

3. Configuration:
   - Run the setup client:
     ```
     python setup_client.py
     ```
   - Follow the prompts to configure cameras, detection areas, and logic conditions
   - The configuration is stored in `obs_config.json`

4. Usage:
   - Start the main script:
     ```
     python main.py
     ```

## Configuration File

The `obs_config.json` file contains the necessary settings for the application. Here's an example of the structure:

```json
{
    "url": "ws://192.168.1.23:4455",
    "password": "placeholder",
    "cameras": {
        "camera1": "http://camera1_ip:port/stream",
        "camera2": "http://camera2_ip:port/stream"
    },
    "logic_conditions": [
        {
            "conditions": [
                {
                    "camera": "camera1",
                    "detection_type": "person",
                    "condition_type": "presence"
                }
            ],
            "scene": "Scene 1"
        }
    ]
}
```

- `url`: The WebSocket URL for your OBS connection
- `password`: Your OBS WebSocket password
- `cameras`: A dictionary of camera names and their MJPEG stream URLs
- `logic_conditions`: An array of condition sets that determine when to switch scenes

## Camera Compatibility

This system is compatible with any camera that can provide an MJPEG stream. This includes:
- USB webcams
- IP cameras
- Smartphones running IP camera apps
- Raspberry Pi with camera module

The flexibility in camera options allows for cost-effective multi-camera setups using existing devices.

## Detection Methods

### Person Detection
The system uses a pre-trained model to detect human presence in camera feeds. This is useful for switching scenes based on where people are located in your streaming setup.

### Motion Detection
Motion detection is used to identify activity in camera feeds, allowing for scene switches based on movement in specific areas. 

Note: The motion detection feature is currently being refined. Improvements are ongoing to increase accuracy and reduce false positives.

## Streamlining Solo Productions

This project aims to simplify the production process for solo content creators:

1. Automated Scene Management: Eliminates the need for manual scene switching during streaming or recording.

2. Multi-Camera Utilization: Enables the use of multiple cameras without additional human operators.

3. Customizable Triggers: Allows for complex scene-switching logic based on activity in different areas of the stream.

4. Cost-Effective Setup: Utilizes MJPEG streaming, allowing for the use of a wide range of camera devices.

5. Reduced Cognitive Load: By handling scene transitions automatically, it allows the content creator to focus on their primary activity or performance.

## Limitations and Considerations

- Relying on lower end wifi cards will result in latency. It is recommended to use a wired connection for the cameras for high fidelity and framerate.
- Do not port forward any cameras. That is a massive security risk.
- Performance depends on the processing power of the host computer and the quality/number of camera streams.
- Motion detection accuracy may vary and is subject to ongoing improvements.

This tool is particularly made for me but is currently being ported to be more modular and user-friendly for the general public. If you have any questions or suggestions, feel free to reach out!
