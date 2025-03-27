from ultralytics import YOLO

models = [
    # "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt"
    ]

for model in models:
    model = YOLO(model)  # load a pretrained model

        # Train the model with memory-saving parameters
    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        project=f"ms-{model}",
        batch=8,  # Reduce batch size (default is 16)
        # cache=False,  # Disable caching
        workers=2  # Reduce number of workers
    )