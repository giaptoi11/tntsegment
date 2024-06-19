from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n-seg.yaml")  # build a new model from YAML
model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolov8n-seg.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="/home/toi/myproject/yolotraining/training.yaml", epochs=1000, imgsz=320,
                      device=0,
                      project= "tnt", name="version",
                      batch=1)