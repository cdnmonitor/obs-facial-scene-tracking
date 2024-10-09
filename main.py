import asyncio
from obs_connection import OBSConnection
from config_loader import load_config
from camera_processing import process_camera_feed

async def main():
    config = load_config()
    obs = OBSConnection()

    await obs.connect()
    await process_camera_feed(obs, config['desktop_camera_url'], config['cnc_camera_url'])
    await obs.disconnect()

if __name__ == '__main__':
    asyncio.run(main())
