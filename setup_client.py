import asyncio
import json
from obs_connection import OBSConnection

class SetupClient:
    def __init__(self):
        self.config = self.load_config()
        self.obs = OBSConnection()

    def load_config(self):
        try:
            with open('obs_config.json', 'r') as config_file:
                config = json.load(config_file)
        except FileNotFoundError:
            config = {}

        # Ensure all required keys are present
        default_config = {
            "url": "ws://localhost:4455",
            "password": "",
            "cameras": {},
            "logic_conditions": []
        }
        
        for key, value in default_config.items():
            if key not in config:
                config[key] = value

        return config

    async def run(self):
        await self.obs.connect()
        scenes = await self.fetch_obs_scenes()
        self.display_scenes(scenes)
        await self.manage_cameras()
        await self.manage_conditions(scenes)
        await self.obs.disconnect()
        self.save_config()

    async def fetch_obs_scenes(self):
        return await self.obs.list_scenes()

    def display_scenes(self, scenes):
        print("Available OBS Scenes:")
        for i, scene in enumerate(scenes, 1):
            print(f"{i}. {scene['sceneName']}")

    async def manage_cameras(self):
        while True:
            print("\nCamera Management:")
            print("1. Add camera")
            print("2. Remove camera")
            print("3. List cameras")
            print("4. Return to Main Menu")
            choice = input("Enter your choice: ")

            if choice == '1':
                name = input("Enter camera name: ")
                url = input("Enter camera URL: ")
                self.config['cameras'][name] = url
            elif choice == '2':
                name = input("Enter camera name to remove: ")
                if name in self.config['cameras']:
                    del self.config['cameras'][name]
                    print(f"Camera '{name}' removed.")
                else:
                    print(f"Camera '{name}' not found.")
            elif choice == '3':
                self.list_cameras()
            elif choice == '4':
                break
            else:
                print("Invalid choice. Please try again.")

    def list_cameras(self):
        print("\nConfigured Cameras:")
        for name, url in self.config['cameras'].items():
            print(f"{name}: {url}")

    async def manage_conditions(self, scenes):
        while True:
            print("\nCondition Management:")
            print("1. Add condition")
            print("2. Remove condition")
            print("3. List conditions")
            print("4. Return to Main Menu")
            choice = input("Enter your choice: ")

            if choice == '1':
                self.add_condition(scenes)
            elif choice == '2':
                self.remove_condition()
            elif choice == '3':
                self.list_conditions()
            elif choice == '4':
                break
            else:
                print("Invalid choice. Please try again.")

    def add_condition(self, scenes):
        print("\nCreate a new condition set:")
        
        conditions = []
        while True:
            print("\nAdding condition:")
            # List available cameras
            print("Available cameras:")
            for i, camera_name in enumerate(self.config['cameras'].keys(), 1):
                print(f"{i}. {camera_name}")
            
            if not self.config['cameras']:
                print("No cameras configured. Please add a camera first.")
                return

            while True:
                camera_choice = input("Enter the number of the camera to use: ")
                try:
                    camera_index = int(camera_choice) - 1
                    if 0 <= camera_index < len(self.config['cameras']):
                        camera = list(self.config['cameras'].keys())[camera_index]
                        break
                    else:
                        print("Invalid camera number. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")

            detection_type = input("Detection type (person/motion): ")
            while detection_type not in ['person', 'motion']:
                print("Invalid detection type. Please enter 'person' or 'motion'.")
                detection_type = input("Detection type (person/motion): ")

            condition_type = input("Condition type (presence/absence): ")
            while condition_type not in ['presence', 'absence']:
                print("Invalid condition type. Please enter 'presence' or 'absence'.")
                condition_type = input("Condition type (presence/absence): ")

            new_condition = {
                "camera": camera,
                "detection_type": detection_type,
                "condition_type": condition_type,
            }

            if conditions:
                print("\nOperator for this condition:")
                print("1. AND")
                print("2. OR")
                operator_choice = input("Choose operator (1/2): ")
                if operator_choice == '1':
                    new_condition["operator"] = 'and'
                elif operator_choice == '2':
                    new_condition["operator"] = 'or'
                else:
                    print("Invalid choice. Defaulting to AND.")
                    new_condition["operator"] = 'and'

            conditions.append(new_condition)

            add_more = input("Add another condition? (y/n): ").lower()
            if add_more != 'y':
                break

        print("\nAvailable scenes:")
        for i, scene in enumerate(scenes, 1):
            print(f"{i}. {scene['sceneName']}")
        scene_index = int(input("Enter the number of the scene to switch to: ")) - 1
        scene = scenes[scene_index]['sceneName']

        self.config['logic_conditions'].append({
            "conditions": conditions,
            "scene": scene
        })
        print("Condition set added successfully.")

    def list_conditions(self):
        print("\nConfigured Condition Sets:")
        for i, condition_set in enumerate(self.config['logic_conditions'], 1):
            print(f"{i}. If ", end="")
            for j, cond in enumerate(condition_set['conditions']):
                if j > 0:
                    print(f" {cond['operator'].upper()} ", end="")
                presence_str = "is" if cond['condition_type'] == 'presence' else "is not"
                print(f"{cond['camera']} {presence_str} detecting {cond['detection_type']}", end="")
            print(f", switch to scene '{condition_set['scene']}'")


    def remove_condition(self):
        self.list_conditions()
        if not self.config['logic_conditions']:
            return
        index = int(input("Enter the number of the condition to remove: ")) - 1
        if 0 <= index < len(self.config['logic_conditions']):
            del self.config['logic_conditions'][index]
            print("Condition removed successfully.")
        else:
            print("Invalid condition number.")
            
    def save_config(self):
        with open('obs_config.json', 'w') as config_file:
            json.dump(self.config, config_file, indent=4)
        print("Configuration saved successfully!")

if __name__ == "__main__":
    client = SetupClient()
    asyncio.run(client.run())