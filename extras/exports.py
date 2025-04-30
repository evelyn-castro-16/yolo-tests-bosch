# from ultralytics import YOLO
#
# # Load a YOLO11n PyTorch model
# model = YOLO(r"C:\Users\Evelyn\Downloads\yolo11n-seg-lane.pt")
#
# # Export the model to NCNN format
# model.export(format="ncnn")  # creates 'yolo11n_ncnn_model'
# # model.export(format="ncnn", half = True)  # creates 'yolo11n_ncnn_model'
#
# # Load the exported NCNN model
# # ncnn_model = YOLO("../yolov8n_ncnn_model")
#
# # Run inference
# # results = ncnn_model(r"D:\0_videos\BOSCH\STOP-DARK.mp4", show= True, conf=0.60)

from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("yolo11n.pt")

# Export the model to TensorRT
model.export(format="engine")  # creates 'yolo11n.engine'

# # Load the exported TensorRT model
# trt_model = YOLO("yolo11n.engine")
#
# # Run inference
# results = trt_model("https://ultralytics.com/images/bus.jpg")