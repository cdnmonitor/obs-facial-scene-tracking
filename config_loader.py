import json

def load_config(config_path='obs_config.json'):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)