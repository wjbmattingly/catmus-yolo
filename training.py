from ultralytics import YOLO

models = [
    # "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt"
    ]

for model_name in models:
    model = YOLO(model_name)  # load a pretrained model

        # Train the model with memory-saving parameters
    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,  # Reduce batch size (default is 16)
        # cache=False,  # Disable caching
        name=f"ms_{model_name.replace('.pt', '')}",
        workers=2  # Reduce number of workers
    )