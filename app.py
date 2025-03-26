import gradio as gr
import supervision as sv
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Load the YOLO model from Hugging Face
model_path = hf_hub_download(
    repo_id="cultural-heritage/medieval-manuscript-yolov11",
    filename="medieval-yolov11n.pt"
)
# Load the YOLO model from local path
model = YOLO(model_path)

model = YOLO
def detect_objects(image, conf_threshold, iou_threshold):
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Perform inference
    results = model(image)
    
    # Extract detections
    detections = sv.Detections.from_yolov5(results)

    # Filter by confidence
    mask = detections.confidence >= conf_threshold
    detections = detections[mask]

    # Create annotator
    box_annotator = sv.BoxAnnotator()
    
    # Annotate image
    annotated_image = box_annotator.annotate(
        scene=image.copy(),
        detections=detections
    )

    return image, annotated_image

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Object Detection with YOLO")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image")
            with gr.Row():
                clear_btn = gr.Button("Clear")
                detect_btn = gr.Button("Detect", variant="primary")
            
            with gr.Row():
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.25,
                )
                
            with gr.Row():
                iou_threshold = gr.Slider(
                    label="NMS IOU Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.45,
                )
                
        with gr.Column():
            output_image = gr.Image(label="Detection Result")

    def clear():
        return None, None

    # Connect buttons to functions
    detect_btn.click(
        detect_objects,
        inputs=[input_image, conf_threshold, iou_threshold],
        outputs=[input_image, output_image]
    )
    clear_btn.click(
        clear,
        inputs=None,
        outputs=[input_image, output_image]
    )

if __name__ == "__main__":
    demo.launch(debug=True)
