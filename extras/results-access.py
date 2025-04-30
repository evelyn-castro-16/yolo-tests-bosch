
from ultralytics import YOLO

# Load your model
# model = YOLO('yolov8n.pt')  # For example, 'yolov8n.pt'
model = YOLO(r"C:\Users\Evelyn\Downloads\yolo11n-seg-lane_ncnn_model")

# source = r"D:\0_videos\BOSCH\SIGNS\smoke-all-signals.mp4"
# source = r"D:\0_videos\stop2.mp4"
source = "http://10.30.100.33:4747/video"
# Run prediction
results = model(source=source, show=True, stream=True, conf = 0.6)

# results = list(model(0))
for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs


        # print(boxes, masks, probs)
        print(boxes)
# Now, iterate over detected objects
# for det in results[0].boxes:
#     # det is now a single detection with attributes you can directly access
#     xmin, ymin, xmax, ymax = det.xyxy[0]  # Coordinates
#     conf = det.conf  # Confidence
#     cls = det.cls  # Class ID
#     print(f"Box coordinates: {xmin}, {ymin}, {xmax}, {ymax}, Confidence: {conf}, Class ID: {cls}")