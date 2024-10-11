import asyncio
from obs_connection import OBSConnection
from camera_processing import process_camera_feeds
from config_loader import load_config

async def main():
    config = load_config()
    
    if 'cameras' not in config or not config['cameras']:
        print("Error: No cameras configured.")
        print("Please run the setup_client.py script to configure your cameras and conditions.")
        return

    obs = OBSConnection()

    try:
        await obs.connect()
        await process_camera_feeds(obs, config)
    finally:
        await obs.disconnect()

if __name__ == '__main__':
    asyncio.run(main())