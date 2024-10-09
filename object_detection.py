import torch

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_objects(frame, confidence_threshold=0.5):
    results = model(frame)
    results = results.pandas().xyxy[0]
    return results[results['confidence'] > confidence_threshold]['name'].values.tolist()
