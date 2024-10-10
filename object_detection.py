import torch
import warnings
from functools import partial

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp.autocast")

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Modify the model's autocast attribute to use the new syntax
if hasattr(model, 'model') and hasattr(model.model, 'autocast'):
    model.model.autocast = partial(torch.amp.autocast, device_type='cuda')

def detect_objects(frame, confidence_threshold=0.5):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        results = model(frame)
    results = results.pandas().xyxy[0]
    return results[results['confidence'] > confidence_threshold]['name'].tolist()