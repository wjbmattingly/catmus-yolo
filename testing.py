from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from typing import Dict

# Define models
MODEL_OPTIONS = {
    "YOLOv11-Nano": "medieval-yolov11n.pt",
    "YOLOv11-Small": "medieval-yolov11s.pt",
    "YOLOv11-Medium": "medieval-yolov11m.pt",
    "YOLOv11-Large": "medieval-yolov11l.pt",
    "YOLOv11-XLarge": "medieval-yolov11x.pt"
}

# Dictionary to store loaded models
models: Dict[str, YOLO] = {}

all_results = {}
# Load all models
for name, model_file in MODEL_OPTIONS.items():
    model_path = hf_hub_download(
        repo_id="biglam/medieval-manuscript-yolov11",
        filename=model_file
    )
    models[name] = YOLO(model_path)
    model = YOLO(model_path)
    metrics = model.val(data="data.yaml", verbose=True)

    cls_metrics = metrics.box.cls
    maps_metrics = metrics.box.maps
    all_results[name] = {
        "cls_metrics": cls_metrics,
        "maps_metrics": maps_metrics
    }

# Create markdown output
with open('results.md', 'w') as f:
    f.write('# YOLOv11 Model Evaluation Results\n\n')
    
    # Write classification metrics
    f.write('## Classification Metrics\n\n')
    f.write('| Model | Precision | Recall | mAP50 | mAP50-95 |\n')
    f.write('|-------|-----------|---------|--------|----------|\n')
    
    for name, results in all_results.items():
        cls = results['cls_metrics']
        maps = results['maps_metrics']
        f.write(f'| {name} | {cls.precision:.3f} | {cls.recall:.3f} | {maps[0]:.3f} | {maps[-1]:.3f} |\n')

print("Results have been written to results.md")
