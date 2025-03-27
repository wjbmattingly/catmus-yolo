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
    
    # Get metrics from the validation results
    results_dict = metrics.results_dict
    all_results[name] = {
        "precision": results_dict['metrics/precision(B)'],
        "recall": results_dict['metrics/recall(B)'],
        "mAP50": results_dict['metrics/mAP50(B)'],
        "mAP50-95": results_dict['metrics/mAP50-95(B)']
    }

# Create markdown output
with open('results.md', 'w') as f:
    f.write('# YOLOv11 Model Evaluation Results\n\n')
    
    # Write classification metrics
    f.write('## Classification Metrics\n\n')
    f.write('| Model | Precision | Recall | mAP50 | mAP50-95 |\n')
    f.write('|-------|-----------|---------|--------|----------|\n')
    
    for name, results in all_results.items():
        f.write(f"| {name} | {results['precision']:.3f} | {results['recall']:.3f} | {results['mAP50']:.3f} | {results['mAP50-95']:.3f} |\n")

print("Results have been written to results.md")
