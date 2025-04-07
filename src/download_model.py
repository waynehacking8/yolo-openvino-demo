import os
from ultralytics import YOLO
from pathlib import Path
import shutil # Import shutil for moving files

def download_yolo_model():
    """Download YOLOv8 pre-trained model and export to different formats"""
    print("Downloading YOLOv8 model...")
    
    # Create model directory
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # Download YOLOv8n model
    # Load the model first (YOLO object initialization might handle download)
    model = YOLO("yolov8n.pt") 
    # The .pt file is usually saved in the working directory or a cache location.
    # Let's assume it's available after initialization for export.
    # We might need to explicitly save/move the base .pt file if needed later.
    
    # --- Exporting ---
    # Ultralytics export often saves to the current working directory 
    # or a 'runs/detect/export' style directory by default.
    # We will export and then move the files to our target 'models' directory.
    
    base_name = "yolov8n" # Base name for exported files
    
    # PyTorch TorchScript
    print("Exporting to PyTorch TorchScript format...")
    try:
        # Remove path, export with default naming
        export_result_ts = model.export(format="torchscript") 
        # export_result_ts likely contains the path to the exported file
        if export_result_ts and 'path' in export_result_ts and os.path.exists(export_result_ts['path']):
            default_ts_path = Path(export_result_ts['path'])
            target_ts_path = model_dir / f"{base_name}_torchscript.pt"
            print(f"Moving {default_ts_path} to {target_ts_path}")
            shutil.move(str(default_ts_path), str(target_ts_path))
        else:
             # Fallback if path isn't returned: Check common default names/locations
             default_ts_path_check = Path(f"{base_name}.torchscript")
             if default_ts_path_check.exists():
                  target_ts_path = model_dir / f"{base_name}_torchscript.pt"
                  print(f"Moving {default_ts_path_check} to {target_ts_path}")
                  shutil.move(str(default_ts_path_check), str(target_ts_path))
             else:
                  print(f"Warning: Could not determine default TorchScript path. File might be in CWD or runs/export.")

    except Exception as e:
        print(f"Error exporting TorchScript: {e}")

    # OpenVINO FP32 (Default)
    print("Exporting to OpenVINO FP32 format...")
    openvino_fp32_target_dir = model_dir / f"{base_name}_openvino_model" # Target directory
    openvino_fp32_target_dir.mkdir(exist_ok=True)
    try:
        # Remove path, export with default naming (usually creates a subdir)
        export_result_ov_fp32 = model.export(format="openvino", half=False) 
        if export_result_ov_fp32 and 'path' in export_result_ov_fp32 and os.path.exists(export_result_ov_fp32['path']):
             default_ov_fp32_path = Path(export_result_ov_fp32['path']) # This is likely the .xml or dir
             # If it's a directory created by export, move its contents or the dir itself
             if default_ov_fp32_path.is_dir():
                  print(f"Moving contents of {default_ov_fp32_path} to {openvino_fp32_target_dir}")
                  # Move contents carefully to avoid overwriting the target dir itself if it already exists
                  for item in default_ov_fp32_path.iterdir():
                       shutil.move(str(item), str(openvino_fp32_target_dir / item.name))
                  # Optionally remove the now empty default directory
                  try:
                       default_ov_fp32_path.rmdir()
                  except OSError:
                       pass # Ignore if not empty or other issues
             # If it's just the xml file (less likely for OV export)
             elif default_ov_fp32_path.is_file() and default_ov_fp32_path.suffix == '.xml':
                  base_ov_name = default_ov_fp32_path.stem
                  default_bin_path = default_ov_fp32_path.with_suffix('.bin')
                  target_xml_path = openvino_fp32_target_dir / f"{base_name}.xml"
                  target_bin_path = openvino_fp32_target_dir / f"{base_name}.bin"
                  print(f"Moving {default_ov_fp32_path} to {target_xml_path}")
                  shutil.move(str(default_ov_fp32_path), str(target_xml_path))
                  if default_bin_path.exists():
                       print(f"Moving {default_bin_path} to {target_bin_path}")
                       shutil.move(str(default_bin_path), str(target_bin_path))
        else:
             # Fallback check: look for default directory name
             default_ov_dir_check = Path(f"{base_name}_openvino_model")
             if default_ov_dir_check.is_dir():
                  print(f"Moving contents of {default_ov_dir_check} to {openvino_fp32_target_dir}")
                  for item in default_ov_dir_check.iterdir():
                      # Ensure target directory exists before moving into it
                      openvino_fp32_target_dir.mkdir(exist_ok=True)
                      shutil.move(str(item), str(openvino_fp32_target_dir / item.name))
                  try:
                       default_ov_dir_check.rmdir()
                  except OSError:
                       pass
             else:
                  print(f"Warning: Could not determine default OpenVINO FP32 path/directory.")

    except Exception as e:
        print(f"Error exporting OpenVINO FP32: {e}")

    # OpenVINO FP16
    print("Exporting to OpenVINO FP16 format...")
    openvino_fp16_target_dir = model_dir / f"{base_name}_openvino_fp16_model" # Target directory
    openvino_fp16_target_dir.mkdir(exist_ok=True)
    try:
        # Remove path, export with default naming + half=True
        export_result_ov_fp16 = model.export(format="openvino", half=True) 
        if export_result_ov_fp16 and 'path' in export_result_ov_fp16 and os.path.exists(export_result_ov_fp16['path']):
             default_ov_fp16_path = Path(export_result_ov_fp16['path'])
             # Assume export creates a directory like <base_name>_openvino_model
             if default_ov_fp16_path.is_dir():
                  print(f"Moving contents of {default_ov_fp16_path} to {openvino_fp16_target_dir}")
                  for item in default_ov_fp16_path.iterdir():
                       # Ensure target directory exists before moving into it
                       openvino_fp16_target_dir.mkdir(exist_ok=True)
                       # Rename files to include _fp16 suffix for clarity
                       target_name = f"{Path(item.stem)}_fp16{item.suffix}" if not item.stem.endswith('_fp16') else item.name
                       shutil.move(str(item), str(openvino_fp16_target_dir / target_name))
                  try:
                       default_ov_fp16_path.rmdir()
                  except OSError:
                       pass
             elif default_ov_fp16_path.is_file() and default_ov_fp16_path.suffix == '.xml':
                  # Less likely, but handle if only XML is returned
                  base_ov_name = default_ov_fp16_path.stem
                  default_bin_path = default_ov_fp16_path.with_suffix('.bin')
                  target_xml_path = openvino_fp16_target_dir / f"{base_name}_fp16.xml"
                  target_bin_path = openvino_fp16_target_dir / f"{base_name}_fp16.bin"
                  print(f"Moving {default_ov_fp16_path} to {target_xml_path}")
                  shutil.move(str(default_ov_fp16_path), str(target_xml_path))
                  if default_bin_path.exists():
                       print(f"Moving {default_bin_path} to {target_bin_path}")
                       shutil.move(str(default_bin_path), str(target_bin_path))

        else:
            # Fallback check: Look for default directory name (_openvino_model is common)
            # The half=True might influence the default name, or might reuse the FP32 dir name
            default_ov_dir_check = Path(f"{base_name}_openvino_model") # Check default name again
            if default_ov_dir_check.is_dir():
                 print(f"Moving contents of {default_ov_dir_check} (assuming FP16 export reused dir) to {openvino_fp16_target_dir}")
                 for item in default_ov_dir_check.iterdir():
                     openvino_fp16_target_dir.mkdir(exist_ok=True)
                     target_name = f"{Path(item.stem)}_fp16{item.suffix}" if not item.stem.endswith('_fp16') else item.name
                     shutil.move(str(item), str(openvino_fp16_target_dir / target_name))
                 try:
                    # Don't remove dir here if it might be reused by ONNX export implicitly
                    # default_ov_dir_check.rmdir()
                    pass
                 except OSError:
                    pass
            else:
                 print(f"Warning: Could not determine default OpenVINO FP16 path/directory.")
                 print("You might need to convert the ONNX model using OpenVINO Model Optimizer manually:")
                 print(f"mo --input_model {model_dir / f'{base_name}.onnx'} --data_type FP16 --output_dir {openvino_fp16_target_dir} --model_name {base_name}_fp16")

    except Exception as e:
        print(f"Error exporting OpenVINO FP16: {e}")
        print("You might need to convert the ONNX model using OpenVINO Model Optimizer manually:")
        print(f"mo --input_model {model_dir / f'{base_name}.onnx'} --data_type FP16 --output_dir {openvino_fp16_target_dir} --model_name {base_name}_fp16")

    # ONNX (for TensorRT)
    print("Exporting to ONNX format (for TensorRT)...")
    try:
        # Remove path, export with default naming
        export_result_onnx = model.export(format="onnx") 
        if export_result_onnx and 'path' in export_result_onnx and os.path.exists(export_result_onnx['path']):
             default_onnx_path = Path(export_result_onnx['path'])
             target_onnx_path = model_dir / f"{base_name}.onnx"
             print(f"Moving {default_onnx_path} to {target_onnx_path}")
             shutil.move(str(default_onnx_path), str(target_onnx_path))
        else:
             default_onnx_path_check = Path(f"{base_name}.onnx")
             if default_onnx_path_check.exists():
                  target_onnx_path = model_dir / f"{base_name}.onnx"
                  print(f"Moving {default_onnx_path_check} to {target_onnx_path}")
                  shutil.move(str(default_onnx_path_check), str(target_onnx_path))
             else:
                  print(f"Warning: Could not determine default ONNX path.")

    except Exception as e:
        print(f"Error exporting ONNX: {e}")
    
    print(f"Models downloaded and exported. Please check '{model_dir}' for the final files.")
    # The original .pt model might be in CWD or cache, not necessarily in model_dir yet.
    # Return the intended path for the base model.
    base_pt_path_final = model_dir / f"{base_name}.pt"
    # Check if the base model was downloaded to CWD and move it
    base_pt_cwd = Path(f"{base_name}.pt")
    if base_pt_cwd.exists() and not base_pt_path_final.exists():
         print(f"Moving base model {base_pt_cwd} to {base_pt_path_final}")
         shutil.move(str(base_pt_cwd), str(base_pt_path_final))

    return base_pt_path_final # Return the expected final path

if __name__ == "__main__":
    download_yolo_model() 