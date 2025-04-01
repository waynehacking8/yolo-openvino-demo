import os
from ultralytics import YOLO
from pathlib import Path

def download_yolo_model():
    """Download YOLOv8 pre-trained model and export to different formats"""
    print("Downloading YOLOv8 model...")
    
    # Create model directory
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # Download YOLOv8n model
    model = YOLO("yolov8n.pt")
    model_path = model_dir / "yolov8n.pt"
    
    # Save in different formats
    # PyTorch
    print("Exporting to PyTorch format...")
    model.export(format="torchscript", path=model_dir / "yolov8n_torchscript.pt")
    
    # OpenVINO
    print("Exporting to OpenVINO format...")
    openvino_path = model_dir / "yolov8n_openvino_model"
    openvino_path.mkdir(exist_ok=True)
    model.export(format="openvino", path=openvino_path / "yolov8n")
    
    # ONNX (for TensorRT)
    print("Exporting to ONNX format (for TensorRT)...")
    model.export(format="onnx", path=model_dir / "yolov8n.onnx")
    
    print(f"Models downloaded and exported to: {model_dir}")
    return model_path

if __name__ == "__main__":
    download_yolo_model() 