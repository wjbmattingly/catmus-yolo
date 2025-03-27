from typing import Tuple, Dict
import gradio as gr
import supervision as sv
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Define models
MODEL_OPTIONS = {
    "YOLOv11-Nano": "medieval-yolov11n.pt",
    "YOLOv11-Large": "medieval-yolov11l.pt"
}

# Dictionary to store loaded models
models: Dict[str, YOLO] = {}

# Load all models
for name, model_file in MODEL_OPTIONS.items():
    model_path = hf_hub_download(
        repo_id="cultural-heritage/medieval-manuscript-yolov11",
        filename=model_file
    )
    models[name] = YOLO(model_path)

# Create annotators
LABEL_ANNOTATOR = sv.LabelAnnotator(text_color=sv.Color.BLACK)
BOX_ANNOTATOR = sv.BoxAnnotator()

def detect_and_annotate(
    image: np.ndarray,
    model_name: str,
    conf_threshold: float,
    iou_threshold: float
) -> np.ndarray:
    # Get the selected model
    model = models[model_name]
    
    # Perform inference
    results = model.predict(
        image,
        conf=conf_threshold,
        iou=iou_threshold
    )[0]
    
    # Convert results to supervision Detections
    boxes = results.boxes.xyxy.cpu().numpy()
    confidence = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    
    # Create Detections object
    detections = sv.Detections(
        xyxy=boxes,
        confidence=confidence,
        class_id=class_ids
    )
    
    # Create labels with confidence scores
    labels = [
        f"{results.names[class_id]} ({conf:.2f})"
        for class_id, conf
        in zip(class_ids, confidence)
    ]

    # Annotate image
    annotated_image = image.copy()
    annotated_image = BOX_ANNOTATOR.annotate(scene=annotated_image, detections=detections)
    annotated_image = LABEL_ANNOTATOR.annotate(scene=annotated_image, detections=detections, labels=labels)
    
    return annotated_image

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Medieval Manuscript Detection with YOLO")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Input Image",
                type='numpy'
            )
            with gr.Accordion("Detection Settings", open=True):
                model_selector = gr.Dropdown(
                    choices=list(MODEL_OPTIONS.keys()),
                    value=list(MODEL_OPTIONS.keys())[0],
                    label="Model",
                    info="Select YOLO model variant"
                )
                with gr.Row():
                    conf_threshold = gr.Slider(
                        label="Confidence Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.25,
                    )
                    iou_threshold = gr.Slider(
                        label="IoU Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.45,
                        info="Decrease for stricter detection, increase for more overlapping boxes"
                    )
            with gr.Row():
                clear_btn = gr.Button("Clear")
                detect_btn = gr.Button("Detect", variant="primary")
                
        with gr.Column():
            output_image = gr.Image(
                label="Detection Result",
                type='numpy'
            )

    def process_image(
        image: np.ndarray,
        model_name: str,
        conf_threshold: float,
        iou_threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        if image is None:
            return None, None
        annotated_image = detect_and_annotate(image, model_name, conf_threshold, iou_threshold)
        return image, annotated_image

    def clear():
        return None, None

    # Connect buttons to functions
    detect_btn.click(
        process_image,
        inputs=[input_image, model_selector, conf_threshold, iou_threshold],
        outputs=[input_image, output_image]
    )
    clear_btn.click(
        clear,
        inputs=None,
        outputs=[input_image, output_image]
    )

if __name__ == "__main__":
    demo.launch(debug=True, show_error=True)
