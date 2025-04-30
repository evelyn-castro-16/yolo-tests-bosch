from ultralytics import YOLO

# Load a model
# model = YOLO("../yolov8n.pt")  # load an official model
# model = YOLO("yolo11n_traffic_objects_V11.pt")  # load an official model
model = YOLO(r"C:\Users\Evelyn\Downloads\yolo11n-seg-lane_ncnn_model")

# source = r"D:\0_videos\BOSCH\SIGNS\smoke-all-signals.mp4"
# source = r"D:\0_videos\stop2.mp4"
source = "http://10.30.100.33:4747/video"
# Predict with the model
results = model(source=source, show=True,
                # stream=True
                )  # predict on an image

for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs

        print("Results: boxes, masks, probs")
        print(boxes, masks, probs)
