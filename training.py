from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # load a pretrained model

# Train the model with memory-saving parameters
results = model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,  # Reduce batch size (default is 16)
    # cache=False,  # Disable caching
    workers=2  # Reduce number of workers
)