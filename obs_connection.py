import simpleobsws
import json

class OBSConnection:
    def __init__(self, config_path='obs_config.json'):
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
            self.url = config['url']
            self.password = config['password']

        self.parameters = simpleobsws.IdentificationParameters()
        self.parameters.rpc_version = 1
        self.ws = simpleobsws.WebSocketClient(url=self.url, password=self.password, identification_parameters=self.parameters)

    async def connect(self):
        try:
            print("[DEBUG] Attempting to connect to OBS WebSocket...")
            await self.ws.connect()
            await self.ws.wait_until_identified()
            print("[DEBUG] Connected to OBS WebSocket and identified successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to connect to OBS WebSocket: {e}")

    async def disconnect(self):
        await self.ws.disconnect()
        print("[DEBUG] Disconnected from OBS WebSocket.")

    async def list_scenes(self):
        try:
            if not self.ws.identified:
                print("[ERROR] Not identified with OBS WebSocket. Cannot make requests.")
                return []

            print("[DEBUG] Requesting scene list from OBS...")
            request = simpleobsws.Request('GetSceneList')
            response = await self.ws.call(request)
            if response.ok():
                scenes = response.responseData['scenes'][::-1]  # Reverse order to match OBS UI
                print(f"[DEBUG] Retrieved {len(scenes)} scenes from OBS.")
                return scenes
            else:
                print(f"[ERROR] Failed to get scene list: {response.responseData}")
                return []
        except Exception as e:
            print(f"[ERROR] Exception occurred while listing scenes: {e}")
            return []

    async def switch_scene(self, scene_name):
        try:
            print(f"[DEBUG] Attempting to switch to scene: {scene_name}")
            request = simpleobsws.Request('SetCurrentProgramScene', {"sceneName": scene_name})
            response = await self.ws.call(request)
            if response.ok():
                print(f"[DEBUG] Successfully switched to scene: {scene_name}")
            else:
                print(f"[ERROR] Failed to switch to scene {scene_name}: {response.responseData}")
        except Exception as e:
            print(f"[ERROR] Exception occurred while switching scene: {e}")

    async def get_current_scene(self):
        try:
            print("[DEBUG] Requesting current scene from OBS...")
            request = simpleobsws.Request('GetCurrentProgramScene')
            response = await self.ws.call(request)
            if response.ok():
                current_scene = response.responseData['currentProgramSceneName']
                print(f"[DEBUG] Current scene is: {current_scene}")
                return current_scene
            else:
                print(f"[ERROR] Failed to get current scene: {response.responseData}")
                return None
        except Exception as e:
            print(f"[ERROR] Exception occurred while getting current scene: {e}")
            return None
