# quantize_nncf.py
import nncf
import openvino as ov
import glob
import numpy as np
import cv2
import os
from pathlib import Path

# --- Configuration ---
FP32_MODEL_XML = Path("models/yolov8n.xml")
CALIBRATION_DATA_DIR = Path("test_images")
INT8_MODEL_DIR = Path("yolov8n_openvino_model_int8")
NUM_CALIBRATION_SAMPLES = 300 # Number of images to use for calibration
INPUT_SIZE = (640, 640) # (height, width) - Should match model input

# --- Helper: Load and Preprocess Image ---
# (Adapted from simple_demo.py, but simplified for calibration)
def preprocess_image_for_calibration(image_path, target_size):
    """Loads and preprocesses a single image for NNCF calibration."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not read image {image_path}, skipping.")
        return None

    # Resize
    h, w = target_size
    img_resized = cv2.resize(img, (w, h))

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1] and change layout from HWC to CHW
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_normalized, (2, 0, 1))

    # Add batch dimension
    img_batch = np.expand_dims(img_chw, axis=0)
    return img_batch

# --- Calibration Dataset Loader for NNCF ---
def create_calibration_dataset(data_dir: Path, num_samples: int, input_name: str, target_size: tuple) -> tuple[nncf.Dataset, int]:
    """Creates an NNCF Dataset object and returns the actual number of items used."""
    image_paths = sorted(list(data_dir.glob("*.jpg")) + \
                         list(data_dir.glob("*.jpeg")) + \
                         list(data_dir.glob("*.png")))

    if not image_paths:
        raise ValueError(f"No images found in {data_dir}")

    selected_paths = image_paths[:min(num_samples, len(image_paths))]
    print(f"Using {len(selected_paths)} images for calibration from {data_dir}")

    def transform_fn(image_path):
        processed_image = preprocess_image_for_calibration(image_path, target_size)
        if processed_image is None: return None
        return {input_name: processed_image}

    dataset_items = [item for path in selected_paths if (item := transform_fn(path)) is not None]

    actual_num_items = len(dataset_items) # Get length here
    if not dataset_items:
         raise ValueError("Could not process any images for the calibration dataset.")

    print(f"Successfully processed {actual_num_items} images for calibration dataset.")
    return nncf.Dataset(dataset_items), actual_num_items # Return count too

# --- Main Quantization Function ---
def main():
    print("--- Starting NNCF INT8 Quantization ---")

    # 1. Load the FP32 OpenVINO model
    print(f"Loading FP32 model from: {FP32_MODEL_XML}")
    if not FP32_MODEL_XML.exists():
        print(f"Error: FP32 model not found at {FP32_MODEL_XML}. "
              f"Please ensure it's generated first (e.g., by exporting from .pt).")
        return
    core = ov.Core()
    model = core.read_model(model=FP32_MODEL_XML)

    # Get the model's input name and expected input size
    try:
         input_node = model.input(0)
         input_name = input_node.get_any_name()
         # Shape is usually NCHW, get H and W
         input_shape = tuple(input_node.shape)
         if len(input_shape) == 4:
              input_height, input_width = input_shape[2], input_shape[3]
              print(f"Model input name: '{input_name}', Expected input size (HxW): {input_height}x{input_width}")
              # Use model's expected size if different from default
              global INPUT_SIZE
              if INPUT_SIZE != (input_height, input_width):
                   print(f"Updating target size for preprocessing to: {input_height}x{input_width}")
                   INPUT_SIZE = (input_height, input_width)
         else:
              print(f"Warning: Unexpected input shape {input_shape}. Using default size {INPUT_SIZE}")
    except Exception as e:
         print(f"Warning: Could not reliably determine input info ({e}). Using default name 'images' and size {INPUT_SIZE}")
         input_name = 'images' # Common default


    # 2. Create the calibration dataset
    print("Creating calibration dataset...")
    actual_num_calibration_items = 0 # Initialize
    try:
        # Get both the dataset and the count
        calibration_dataset, actual_num_calibration_items = create_calibration_dataset(
            CALIBRATION_DATA_DIR, NUM_CALIBRATION_SAMPLES, input_name, INPUT_SIZE
        )
    except Exception as e:
        print(f"Error creating calibration dataset: {e}")
        return

    if actual_num_calibration_items == 0:
         print("Error: No calibration data could be processed.")
         return

    # 3. Perform quantization
    print("Starting quantization process (this may take some time)...")
    # Use the actual count for subset_size
    quantized_model = nncf.quantize(
        model,
        calibration_dataset,
        subset_size=actual_num_calibration_items # Use the actual number of items
    )
    print("Quantization finished.")

    # 4. Save the INT8 model
    INT8_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    int8_xml_path = INT8_MODEL_DIR / f"{FP32_MODEL_XML.stem}_int8.xml" # e.g., yolov8n_int8.xml
    print(f"Saving INT8 model to: {int8_xml_path}")
    ov.save_model(quantized_model, int8_xml_path)

    print("--- NNCF INT8 Quantization Complete ---")

if __name__ == "__main__":
    main() 