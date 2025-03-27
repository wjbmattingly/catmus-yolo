import os
import argparse
from PIL import Image
import yaml
import numpy as np

from datasets import load_dataset

# Default dataset (kept for backward compatibility)
DEFAULT_DATASET = "CATMuS/medieval-segmentation"

Image.MAX_IMAGE_PIXELS = None


def resize_with_aspect_ratio(image, max_size=1500):
    """Resize image maintaining aspect ratio so largest dimension is max_size"""
    # Convert to RGB if image is in RGBA mode
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    width, height = image.size
    if width > height:
        if width > max_size:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            return image, 1.0
    else:
        if height > max_size:
            new_height = max_size
            new_width = int(width * (max_size / height))
        else:
            return image, 1.0
    
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    scale_factor = new_width / width
    return resized_image, scale_factor

def update_yaml_file(yaml_path, new_classes, class_source):
    """Update data.yaml file with new classes while preserving existing ones"""
    import yaml
    
    # Read existing yaml if it exists
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        existing_names = yaml_data.get('names', [])
        if isinstance(existing_names, list):
            existing_names = existing_names
        else:
            existing_names = []
    else:
        yaml_data = {
            'train': '../train/images',
            'val': '../valid/images',
            'names': []
        }
        existing_names = []

    # Add new classes while preserving existing ones
    updated_names = existing_names.copy()
    for class_name in new_classes:
        if class_name not in updated_names:
            updated_names.append(class_name)
    
    # Create the yaml content in the exact required format
    yaml_content = f"""train: ../train/images
val: ../valid/images

nc: {len(updated_names)}
names: {updated_names}"""
    
    # Write the formatted string directly to file
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    return updated_names

def sanitize_filename(filename):
    """
    Convert filename to a safe version that works on all operating systems
    """
    # Replace problematic characters with underscores
    invalid_chars = '<>:"/\\|?*., '
    
    # First replace slashes with dashes to maintain some readability
    filename = filename.replace('/', '-').replace('\\', '-')
    
    # Then replace other invalid characters with underscore
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove any duplicate underscores
    while '__' in filename:
        filename = filename.replace('__', '_')
    
    # Remove leading/trailing underscores
    filename = filename.strip('_')
    
    return filename

def has_polygon_annotations(objects):
    """Check if the dataset contains polygon annotations"""
    # Check for common polygon/mask keys
    polygon_keys = ['segmentation', 'polygons', 'mask', 'masks']
    return any(key in objects for key in polygon_keys)

def convert_polygon_to_yolo(polygon, image_width, image_height):
    """Convert polygon coordinates to YOLO format (normalized)"""
    # Flatten the polygon list and normalize coordinates
    yolo_poly = []
    for i in range(0, len(polygon), 2):
        x = polygon[i] / image_width
        y = polygon[i+1] / image_height
        yolo_poly.extend([x, y])
    return yolo_poly

def convert_to_yolov11_format(data, output_dir="./", split="train", create_yaml=True, class_source='category', max_size=1500, shelfmark_counts=None, segmentation_type='bbox'):
    """
    Converts the given input dictionary to YOLOv11 PyTorch TXT format.  Allows
    specifying whether to use 'category' or 'type' for class labels.

    Args:
        data (dict): A dictionary containing image and annotation data.
                     See the original function for the full structure.
        output_dir (str, optional): Output directory. Defaults to "yolov11_output".
        split (str, optional): Which split to save to ("train" or "valid")
        create_yaml (bool, optional): Whether to create data.yaml. Defaults to True.
        class_source (str, optional):  Which field to use for class labels.
                                       Must be 'category' or 'type'.
                                       Defaults to 'category'.
        max_size (int): Maximum dimension for image resizing
        shelfmark_counts (dict, optional): Dictionary to track duplicates
        segmentation_type (str): Type of segmentation to use ('bbox' or 'mask')

    Returns:
        bool: True if conversion was successful, False otherwise.

    Raises:
        TypeError: If 'image' is present and is not a PIL Image.
        ValueError: If 'bbox' coordinates are invalid.
        ValueError: If class_source is not 'category' or 'type'.
        KeyError: If required keys are missing.
    """

    # Input validation (including class_source)
    if class_source not in ('category', 'type'):
        raise ValueError("class_source must be 'category' or 'type'")
    
    if segmentation_type not in ('bbox', 'mask'):
        raise ValueError("segmentation_type must be 'bbox' or 'mask'")

    required_keys = ['width', 'height', 'objects', 'shelfmark', 'century', 'project']
    if not all(key in data for key in required_keys):
        print(f"Skipping: Missing required keys for {data.get('shelfmark', 'unknown')}")
        return False

    objects_required_keys = ['id', 'bbox', 'category', 'type', 'parent']  # baseline is optional.
    if not all(key in data['objects'] for key in objects_required_keys):
        raise KeyError(f"Missing required keys in input data['objects']. Required: {objects_required_keys}")
        
    # Check if polygon/mask annotations are available
    has_polygons = has_polygon_annotations(data['objects'])
    
    # If mask segmentation requested but no polygons available, fall back to bbox
    if segmentation_type == 'mask' and not has_polygons:
        print(f"Warning: Mask segmentation requested but no polygon annotations found. Falling back to bounding boxes.")
        segmentation_type = 'bbox'

    # Validate all bounding boxes before processing (always needed as fallback)
    image_width = data['width']
    image_height = data['height']
    objects = data['objects']
    
    # Check if any bounding box is invalid (only when using bboxes or as polygon fallback)
    for i in range(len(objects['id'])):
        bbox = objects['bbox'][i]
        x1, y1, x2, y2 = bbox
        if not (0 <= x1 < image_width and 0 <= y1 < image_height and 
                0 <= x2 <= image_width and 0 <= y2 <= image_height and 
                x1 < x2 and y1 < y2):
            print(f"Skipping {data['shelfmark']}: Invalid bounding box {bbox}")
            return False

    # Create the output directory structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

    # Process image if present
    if 'image' in data and data['image'] is not None:
        original_image = data['image']
        # Convert to RGB if needed
        if original_image.mode == 'RGBA':
            original_image = original_image.convert('RGB')
        
        resized_image, scale_factor = resize_with_aspect_ratio(original_image, max_size)
        new_width, new_height = resized_image.size
    else:
        print(f"Skipping {data['shelfmark']}: No image data")
        return False

    # Handle yaml file updates
    if create_yaml:
        yaml_path = os.path.join(output_dir, "data.yaml")
        # Get unique class names from the current data point
        current_classes = []
        for class_name in data['objects'][class_source]:
            if class_name not in current_classes:
                current_classes.append(class_name)
        
        # Update yaml and get back complete list of classes
        all_classes = update_yaml_file(yaml_path, current_classes, class_source)
        
        # Update class_to_id mapping based on position in all_classes
        class_to_id = {name: idx for idx, name in enumerate(all_classes)}

    # Handle shelfmark counting and filename generation
    shelfmark = sanitize_filename(data['shelfmark'])
    if shelfmark_counts is None:
        shelfmark_counts = {}
    
    if shelfmark in shelfmark_counts:
        shelfmark_counts[shelfmark] += 1
        filename_base = f"{shelfmark}_{shelfmark_counts[shelfmark]:05d}"
    else:
        shelfmark_counts[shelfmark] = 0
        filename_base = f"{shelfmark}_00000"

    # Update file paths with new naming scheme
    txt_filename = f"{filename_base}.txt"
    image_filename = f"{filename_base}.jpg"
    
    txt_filepath = os.path.join(output_dir, split, "labels", txt_filename)
    image_filepath = os.path.join(output_dir, split, "images", image_filename)

    # Create annotation file
    with open(txt_filepath, "w") as f:
        for i in range(len(objects['id'])):
            class_id = class_to_id[objects[class_source][i]]
            
            if segmentation_type == 'mask' and has_polygons:
                # Handle polygon/mask segmentation
                polygon = None
                
                # Try different possible polygon/mask keys
                if 'segmentation' in objects and i < len(objects['segmentation']):
                    polygon = objects['segmentation'][i]
                elif 'polygons' in objects and i < len(objects['polygons']):
                    polygon = objects['polygons'][i]
                elif 'mask' in objects and i < len(objects['mask']):
                    # Convert mask to polygon
                    mask = objects['mask'][i]
                    if isinstance(mask, np.ndarray) or hasattr(mask, 'shape'):  # numpy array or similar
                        from skimage.measure import find_contours
                        contours = find_contours(mask, 0.5)
                        if contours:
                            # Take the largest contour
                            largest_contour = max(contours, key=lambda x: len(x))
                            # Convert to flat list [x1,y1,x2,y2,...]
                            polygon = largest_contour.flatten().tolist()
                
                if polygon:
                    # Scale polygon coordinates
                    scaled_polygon = []
                    for j in range(0, len(polygon), 2):
                        if j+1 < len(polygon):
                            scaled_polygon.extend([
                                polygon[j] * scale_factor,
                                polygon[j+1] * scale_factor
                            ])
                    
                    # Convert to YOLO format
                    yolo_poly = convert_polygon_to_yolo(scaled_polygon, new_width, new_height)
                    if yolo_poly:
                        # Write in YOLO format: class_id x1 y1 x2 y2 ...
                        poly_str = ' '.join([f"{coord:.6f}" for coord in yolo_poly])
                        f.write(f"{class_id} {poly_str}\n")
                        continue
                
                # Fallback to bbox if polygon conversion failed
                print(f"Warning: Couldn't convert polygon for object {i}, falling back to bbox")
            
            # Default to bounding box (either by choice or as fallback)
            bbox = objects['bbox'][i]
            x1, y1, x2, y2 = bbox
            
            # Scale coordinates to match resized image
            x1 *= scale_factor
            y1 *= scale_factor
            x2 *= scale_factor
            y2 *= scale_factor

            # Convert to YOLO format (normalized)
            center_x = ((x1 + x2) / 2) / new_width
            center_y = ((y1 + y2) / 2) / new_height
            width = (x2 - x1) / new_width
            height = (y2 - y1) / new_height

            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

    # Save resized image
    resized_image.save(image_filepath, "JPEG", quality=95)

    return True

def process_dataset(dataset, output_dir="./", train_split=0.8, max_size=1500, segmentation_type='bbox'):
    """
    Process the entire dataset and split into train/valid sets
    
    Args:
        dataset: The loaded dataset
        output_dir (str): Output directory
        train_split (float): Proportion of data to use for training (0-1)
        max_size (int): Maximum dimension for image resizing
        segmentation_type (str): Type of segmentation to use ('bbox' or 'mask')
    """
    from tqdm import tqdm
    
    # Dictionary to keep track of shelfmark counts
    shelfmark_counts = {}
    
    # Calculate split indices
    total_size = len(dataset['train'])
    train_size = int(total_size * train_split)
    
    successful_train = 0
    successful_valid = 0
    
    # Process training data
    for i in tqdm(range(train_size), desc="Processing training data"):
        if convert_to_yolov11_format(dataset['train'][i], 
                                   output_dir=output_dir, 
                                   split="train", 
                                   max_size=max_size,
                                   shelfmark_counts=shelfmark_counts,
                                   segmentation_type=segmentation_type):
            successful_train += 1
            
    # Process validation data
    for i in tqdm(range(train_size, total_size), desc="Processing validation data"):
        if convert_to_yolov11_format(dataset['train'][i], 
                                   output_dir=output_dir, 
                                   split="valid", 
                                   max_size=max_size,
                                   shelfmark_counts=shelfmark_counts,
                                   segmentation_type=segmentation_type):
            successful_valid += 1
    
    print(f"\nProcessing complete:")
    print(f"Training: {successful_train}/{train_size} images processed successfully")
    print(f"Validation: {successful_valid}/{total_size-train_size} images processed successfully")

def main():
    parser = argparse.ArgumentParser(description='Convert dataset to YOLOv11 format')
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET,
                       help='HuggingFace dataset name (default: CATMuS/medieval-segmentation)')
    parser.add_argument('--output_dir', type=str, default='./yolov11_output',
                       help='Output directory (default: ./yolov11_output)')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Train split ratio (default: 0.8)')
    parser.add_argument('--max_size', type=int, default=1500,
                       help='Maximum image dimension (default: 1500)')
    parser.add_argument('--seg', type=str, default='bbox',
                       choices=['bbox', 'mask'],
                       help='Segmentation type to use (bbox or mask, default: bbox)')
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = load_dataset(args.dataset)
    
    # Process dataset
    process_dataset(
        dataset=dataset,
        output_dir=args.output_dir,
        train_split=args.train_split,
        max_size=args.max_size,
        segmentation_type=args.seg
    )

if __name__ == '__main__':
    main()
