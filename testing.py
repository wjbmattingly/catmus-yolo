from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from typing import Dict
import yaml

# Load class names from data.yaml
with open('data.yaml', 'r') as f:
    data_config = yaml.safe_load(f)
    class_names = data_config['names']

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
    
    # Get metrics from the validation results
    all_results[name] = {
        "map": metrics.box.map,    # mAP50-95
        "map50": metrics.box.map50,  # mAP50
        "map75": metrics.box.map75,  # mAP75
        "maps": metrics.box.maps     # list of mAP50-95 for each category
    }

# Create markdown output
with open('results.md', 'w') as f:
    f.write('# YOLOv11 Model Evaluation Results\n\n')
    
    # Write overall metrics
    f.write('## Overall Metrics\n\n')
    f.write('| Model | mAP50-95 | mAP50 | mAP75 |\n')
    f.write('|-------|-----------|--------|--------|\n')
    
    for name, results in all_results.items():
        f.write(f"| {name} | {results['map']:.3f} | {results['map50']:.3f} | {results['map75']:.3f} |\n")
    
    # Write per-class metrics for each model
    f.write('\n## Per-Class Metrics\n\n')
    
    for name, results in all_results.items():
        f.write(f'\n### {name}\n\n')
        f.write('| Class | mAP50-95 |\n')
        f.write('|-------|----------|\n')
        
        for class_id, map_value in enumerate(results['maps']):
            class_name = class_names[class_id]
            f.write(f"| {class_name} | {map_value:.3f} |\n")

print("Results have been written to results.md")
