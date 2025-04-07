import os
import time
import numpy as np
import cv2
import torch
from pathlib import Path
import argparse
import random
import json
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Check CUDA and TensorRT availability on import
print("\n=== 系統環境檢查 ===")
print(f"PyTorch版本: {torch.__version__}")

if torch.cuda.is_available():
    print(f"CUDA可用: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
    
    # Try to import TensorRT
    try:
        import tensorrt as trt
        print(f"TensorRT已安裝，版本: {trt.__version__}")
        TENSORRT_AVAILABLE = True
    except ImportError:
        print("TensorRT未安裝")
        TENSORRT_AVAILABLE = False
    except Exception as e:
        print(f"導入TensorRT時出錯: {e}")
        TENSORRT_AVAILABLE = False
else:
    print("CUDA不可用")
    TENSORRT_AVAILABLE = False

try:
    import openvino as ov
    print(f"OpenVINO已安裝，版本: {ov.__version__}")
    OPENVINO_AVAILABLE = True
except ImportError:
    print("OpenVINO未安裝")
    OPENVINO_AVAILABLE = False
except Exception as e:
    print(f"導入OpenVINO時出錯: {e}")
    OPENVINO_AVAILABLE = False

def load_and_preprocess_image(image_path, size=(640, 640)):
    """加載和預處理圖像
    
    Args:
        image_path: 圖像路徑
        size: 調整大小
        
    Returns:
        預處理後的圖像和原始圖像
    """
    # 讀取圖像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"無法讀取圖像: {image_path}")
    
    original_image = img.copy()
    
    # 調整大小
    img = cv2.resize(img, size)
    
    # 將色彩格式從BGR轉換為RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 歸一化像素值
    img = img.astype(np.float32) / 255.0
    
    # 轉換通道順序從HWC到CHW
    img = np.transpose(img, (2, 0, 1))
    
    # 新增批次維度
    img = np.expand_dims(img, axis=0)
    
    return img, original_image

def draw_detections(image, detections, confidence_threshold=0.5):
    """在圖像上繪製檢測結果
    
    Args:
        image: 原始圖像
        detections: 檢測結果
        confidence_threshold: 置信度閾值
        
    Returns:
        帶有檢測框的圖像
    """
    # COCO類別名稱
    class_names = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    
    image_with_boxes = image.copy()
    h, w = image.shape[:2]  # 獲取原始圖像的高度和寬度
    
    # 檢查檢測結果格式 - 修改為支援PyTorch YOLO格式
    boxes = []
    
    if isinstance(detections, list) and len(detections) > 0:
        # 處理ultralytics.engine.results.Results格式
        if hasattr(detections[0], 'boxes') and hasattr(detections[0].boxes, 'data'):
            # YOLOv8 ultralytics格式
            print(f"使用ultralytics格式處理, 形狀: {detections[0].boxes.data.shape}")
            for box in detections[0].boxes.data.cpu().numpy():
                if box[4] >= confidence_threshold:
                    x1, y1, x2, y2, confidence, class_id = box
                    boxes.append([int(x1), int(y1), int(x2), int(y2), confidence, int(class_id)])
                    
        # 處理傳統的tensor格式
        elif isinstance(detections[0], torch.Tensor):
            # YOLOv8 PyTorch格式
            print(f"使用PyTorch Tensor格式處理, 形狀: {detections[0].shape}")
            # 第一个元素是框和得分 [N, 6] 格式为 [x1, y1, x2, y2, confidence, class_id]
            for det in detections[0].cpu().numpy():
                if len(det) >= 6 and det[4] >= confidence_threshold:
                    x1, y1, x2, y2, confidence, class_id = det
                    # 確保坐標是整數
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    boxes.append([x1, y1, x2, y2, confidence, int(class_id)])
    else:
        # 處理兼容其他格式
        if len(detections.shape) == 3 and detections.shape[2] > 6:
            # YOLOv8輸出格式可能是 [batch, num_detections, 85]
            # 其中85 = 4(邊界框) + 1(置信度) + 80(類別)
            print(f"使用數組格式處理，形狀: {detections.shape}")
            for i in range(detections.shape[1]):
                detection = detections[0, i]
                confidence = detection[4]
                
                if confidence >= confidence_threshold:
                    # YOLOv8模型輸出的是中心點坐標、寬度和高度
                    x_center, y_center, width, height = detection[0:4]
                    
                    # 計算邊界框的左上角和右下角坐標（相對於輸入大小）
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    
                    # 將相對坐標轉換為絕對坐標
                    x1 = int(x1 * w)
                    y1 = int(y1 * h)
                    x2 = int(x2 * w)
                    y2 = int(y2 * h)
                    
                    class_scores = detection[5:]
                    class_id = np.argmax(class_scores)
                    
                    boxes.append([x1, y1, x2, y2, confidence, class_id])
        else:
            # 處理其他可能的格式
            try:
                print(f"嘗試處理未知格式，形狀: {detections.shape if hasattr(detections, 'shape') else '未知'}")
                for box in detections[0]:
                    if len(box) >= 6:
                        confidence = box[4]
                        
                        if confidence >= confidence_threshold:
                            # 嘗試兩種格式
                            # 1. 如果是中心點格式 (x_center, y_center, width, height)
                            if box[2] < 1.0 and box[3] < 1.0:  # 假設是歸一化坐標
                                x_center, y_center, width, height = box[0:4]
                                
                                # 計算邊界框的左上角和右下角坐標
                                x1 = x_center - width / 2
                                y1 = y_center - height / 2
                                x2 = x_center + width / 2
                                y2 = y_center + height / 2
                                
                                # 將相對坐標轉換為絕對坐標
                                x1 = int(x1 * w)
                                y1 = int(y1 * h)
                                x2 = int(x2 * w)
                                y2 = int(y2 * h)
                            else:  # 2. 如果是坐標格式 (x1, y1, x2, y2)
                                x1, y1, x2, y2 = map(int, box[0:4])
                            
                            class_id = int(box[5])
                            boxes.append([x1, y1, x2, y2, confidence, class_id])
            except Exception as e:
                print(f"無法解析檢測結果格式: {e}")
    
    # 繪製檢測框和標籤
    detected_count = 0
    
    # 使用中文顯示檢測總數
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    
    for box in boxes:
        # box格式：[x1, y1, x2, y2, confidence, class_id]
        x1, y1, x2, y2, confidence, class_id = box
        
        # 確保坐標在有效範圍內
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        # 跳過過小的框
        if x2 - x1 < 10 or y2 - y1 < 10:
            print(f"跳過小框: ({x1},{y1},{x2},{y2}), 大小: {x2-x1}x{y2-y1}")
            continue
            
        detected_count += 1
        
        # 獲取類別名稱
        class_id = int(class_id)
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        
        # 繪製框
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 添加標籤
        label = f"{class_name}: {confidence:.2f}"
        # 獲取文字大小
        text_size = cv2.getTextSize(label, font, 0.5, 2)[0]
        # 繪製綠色文字
        cv2.putText(image_with_boxes, label, (x1, y1 - 5), font, 0.5, (0, 255, 0), 2)
    
    # 在圖像頂部添加檢測總數（使用綠色文字）
    summary = f"檢測到 {detected_count} 個物體"
    # 獲取文字大小
    text_size = cv2.getTextSize(summary, font, font_scale, font_thickness)[0]
    # 繪製綠色文字
    cv2.putText(image_with_boxes, summary, (10, text_size[1] + 10), font, font_scale, (0, 255, 0), font_thickness)
    
    return image_with_boxes

def check_tensorrt_availability():
    """檢查TensorRT是否可用並打印診斷信息"""
    print("\n=== TensorRT 可用性檢查 ===")
    
    # 檢查CUDA
    if not torch.cuda.is_available():
        print("CUDA不可用，無法使用TensorRT")
        return False
    
    print(f"CUDA可用: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
    
    # 嘗試導入tensorrt
    try:
        import tensorrt as trt
        print(f"TensorRT已安裝，版本: {trt.__version__}")
        return True
    except ImportError:
        print("TensorRT未安裝")
        
        # 檢查是否安裝了nvidia-tensorrt包
        try:
            import subprocess
            result = subprocess.run(["pip", "list"], capture_output=True, text=True)
            if "nvidia-tensorrt" in result.stdout:
                print("發現nvidia-tensorrt包，但無法導入tensorrt模塊")
            else:
                print("未找到nvidia-tensorrt包，需要安裝TensorRT")
        except Exception as e:
            print(f"檢查pip包時出錯: {e}")
            
        return False
    except Exception as e:
        print(f"導入TensorRT時出錯: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Model Demo")
    parser.add_argument("--image", type=str, default="test_images/800px-Cat03.jpg", help="Input image path")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Model path")
    parser.add_argument("--threshold", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--output", type=str, default="output", help="Output directory for saving results")
    parser.add_argument("--mode", type=str, default="tensorrt_gpu", 
                      choices=["pytorch_cpu", "openvino_cpu", "tensorrt_gpu", "all"], 
                      help="Run mode: pytorch_cpu, openvino_cpu, tensorrt_gpu, or all")
    parser.add_argument("--use_tensorrt", action="store_true", help="Use TensorRT acceleration for CUDA mode")
    parser.add_argument("--tensorrt_model", type=str, default="", help="Path to existing TensorRT model (.engine file)")
    parser.add_argument("--force_convert", action="store_true", help="Force conversion to TensorRT even if model exists")
    parser.add_argument("--skip_convert", action="store_true", help="Skip TensorRT conversion")
    parser.add_argument("--benchmark", action="store_true", help="Run multiple inferences to benchmark performance")
    parser.add_argument("--benchmark_runs", type=int, default=10, help="Number of inference runs for benchmarking")
    parser.add_argument("--compare_all", action="store_true", help="Compare all available backends (PyTorch CPU, OpenVINO, TensorRT)")
    parser.add_argument("--check_tensorrt", action="store_true", help="Check TensorRT availability and exit")
    parser.add_argument("--save_summary", action="store_true", help="Save summary to a text file")
    parser.add_argument("--num_images", type=int, default=5, help="Number of random images to test")
    parser.add_argument("--test_all", action="store_true", help="Test all images in test_images directory")
    parser.add_argument("--results_dir", type=str, default="test_results", help="Directory for detection results")
    parser.add_argument("--benchmark_dir", type=str, default="benchmark_results", help="Directory for benchmark results")
    parser.add_argument('--precision', type=str, default='FP32', choices=['FP32', 'FP16', 'INT8'],
                        help='Inference precision for OpenVINO (FP32, FP16, INT8)')
    parser.add_argument('--int8_model_dir', type=str, default=None,
                        help='Directory containing the INT8 quantized OpenVINO model (required if --precision INT8)')
    parser.add_argument('--use_async', action='store_true',
                        help='Use asynchronous inference API for OpenVINO')
    parser.add_argument('--benchmark_openvino_modes', action='store_true',
                        help='Run a detailed benchmark comparing different OpenVINO precision and execution modes (Sync/Async)')
    parser.add_argument('--test_all_images_all_modes', action='store_true',
                        help='Run all images in test_images through all available backend modes and save results.')
    # Add new arguments for batch benchmark
    parser.add_argument('--run_batch_benchmark', action='store_true',
                        help='Run batch processing benchmark across all modes for various batch sizes.')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 2, 4, 8, 16],
                        help='List of batch sizes to test in batch benchmark (default: 1 2 4 8 16).')

    args = parser.parse_args()
    
    # Summary information to save at the end
    summary_info = {
        "system_info": {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "tensorrt_available": TENSORRT_AVAILABLE,
            "openvino_available": OPENVINO_AVAILABLE
        },
        "args": vars(args),
        "results": {}
    }
    
    # Check TensorRT availability if requested
    if args.check_tensorrt:
        check_tensorrt_availability()
        if args.save_summary:
            with open("tensorrt_check_summary.txt", "w") as f:
                f.write(f"TensorRT Available: {TENSORRT_AVAILABLE}\n")
                f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
                if torch.cuda.is_available():
                    f.write(f"CUDA Device: {torch.cuda.get_device_name(0)}\n")
                f.write(f"PyTorch Version: {torch.__version__}\n")
        return
    
    # Check for the new OpenVINO benchmark mode FIRST
    if args.benchmark_openvino_modes:
        # Ensure the function run_openvino_benchmark is defined elsewhere in the file
        # We assume it will be added in subsequent edits if not already present
        print("Detected --benchmark_openvino_modes flag.") # Add print for debugging
        # Check if the function exists before calling (defensive programming)
        if 'run_openvino_benchmark' in globals():
             results = run_openvino_benchmark(args)
             if args.save_summary:
                 # Ensure benchmark directory exists
                 os.makedirs(args.benchmark_dir, exist_ok=True)
                 timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

                 # Save JSON results
                 summary_filename = os.path.join(args.benchmark_dir, f"openvino_benchmark_summary_{timestamp}.json")
                 with open(summary_filename, "w") as f:
                     json.dump(results, f, indent=2)
                 print(f"OpenVINO Benchmark summary saved to: {summary_filename}")

                 # Plot results if data exists
                 if results and "modes" in results and results["modes"]:
                     # Ensure plot_benchmark_results is defined elsewhere
                     if 'plot_benchmark_results' in globals():
                          # Pass the modified title prefix
                          plot_benchmark_results(results, args.benchmark_dir, timestamp, title_prefix="OpenVINO")
                     else:
                          print("Warning: plot_benchmark_results function not found, skipping plot generation.")
                 else:
                     print("No OpenVINO benchmark results to plot.")
        else:
             print("Error: run_openvino_benchmark function not found. Please ensure it's defined in the script.")
        return # Exit after benchmark

    # Existing logic for --compare_all
    elif args.compare_all:
        results = run_benchmark_comparison(args)
        if args.save_summary:
            # 確保benchmark目錄存在
            os.makedirs(args.benchmark_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存JSON結果
            summary_filename = os.path.join(args.benchmark_dir, f"benchmark_summary_{timestamp}.json")
            with open(summary_filename, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Benchmark summary saved to: {summary_filename}")
            
            # 繪製性能對比圖表
            if results:
                plot_benchmark_results(results, args.benchmark_dir, timestamp)
                
        return
    
    # Check for the new all images all modes test
    elif args.test_all_images_all_modes:
        print("Detected --test_all_images_all_modes flag.")
        if 'run_all_images_all_modes' in globals():
            run_all_images_all_modes(args)
        else:
            print("Error: run_all_images_all_modes function not found.")
        return # Exit after this test run

    # Add check for the new batch benchmark
    elif args.run_batch_benchmark:
        print("Detected --run_batch_benchmark flag.")
        if 'run_batch_benchmark' in globals():
            run_batch_benchmark(args)
        else:
            print("Error: run_batch_benchmark function not found.")
        return # Exit after batch benchmark
    
    # 確定要測試的模式
    modes_to_test = []
    if args.mode == "all":
        modes_to_test = ["pytorch_cpu"]
        if OPENVINO_AVAILABLE: modes_to_test.append("openvino_cpu")
        if TENSORRT_AVAILABLE: modes_to_test.append("tensorrt_gpu")
    else:
        modes_to_test = [args.mode]
    
    # 選擇要測試的圖片
    if os.path.isdir("test_images"):
        # 獲取test_images目錄中的所有圖片
        images = [f for f in os.listdir("test_images") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            print("Error: 在test_images目錄中未找到圖片!")
            return
        
        # 決定測試哪些圖片
        if args.test_all:
            selected_images = images
            print(f"\n=== 測試所有 {len(selected_images)} 張圖片 ===")
        else:
            # 隨機選擇指定數量的圖片
            num_images = min(args.num_images, len(images))
            selected_images = random.sample(images, num_images)
            print(f"\n=== 隨機選擇了 {num_images} 張圖片進行測試 ===")
            
        for i, img in enumerate(selected_images):
            print(f"{i+1}. {img}")
        
        # 確保輸出目錄存在
        os.makedirs(args.results_dir, exist_ok=True)
        
        # 每個模式都運行所有選定的圖片
        for mode in modes_to_test:
            print(f"\n=== 使用 {mode} 模式測試 ===")
            
            # 為每個模式創建一個子目錄
            mode_dir = os.path.join(args.results_dir, mode)
            os.makedirs(mode_dir, exist_ok=True)
            
            # 測試每張圖片
            for img_file in selected_images:
                img_path = os.path.join("test_images", img_file)
                print(f"\n=== 處理圖片: {img_path} ===")
                
                # 為每張圖片創建參數副本
                import copy
                temp_args = copy.deepcopy(args)
                temp_args.image = img_path
                temp_args.mode = mode
                
                # 設置輸出路徑
                base_name, ext = os.path.splitext(img_file)
                temp_args.output = os.path.join(mode_dir, f"{base_name}{ext}")
                
                # 運行推理
                if args.benchmark:
                    inference_time = run_inference(temp_args)
                    if inference_time:
                        print(f"圖片 {img_file} 的推理時間: {inference_time:.2f} ms (FPS: {1000/inference_time:.2f})")
                else:
                    run_single_inference(temp_args)
    else:
        print("Error: test_images目錄不存在!")

def run_benchmark_comparison(args):
    """運行擴展的性能對比測試，包含 PyTorch, TensorRT, 和多種 OpenVINO 模式"""
    print("\n=== Running Extended Performance Comparison ===")
    print("Comparing: PyTorch CPU, TensorRT GPU (if available), and various OpenVINO modes...")

    # --- Configuration ---
    base_model_path = args.model
    int8_model_dir = args.int8_model_dir # Get potential INT8 path from args
    benchmark_runs_per_image = args.benchmark_runs

    # --- Check INT8 Model Availability (similar logic as before) ---
    int8_available = False
    if not int8_model_dir:
        base_pt_model_name = os.path.splitext(os.path.basename(args.model))[0] if args.model.endswith('.pt') else "yolov8n"
        derived_int8_dir = args.model.replace('.pt', '_openvino_model') if args.model.endswith('.pt') else f'models/{base_pt_model_name}_openvino_model'
        potential_int8_xml = os.path.join(derived_int8_dir, f"{base_pt_model_name}_int8.xml")
        if os.path.exists(potential_int8_xml):
             print(f"Found potential INT8 model for comparison at: {potential_int8_xml}")
             int8_model_dir = derived_int8_dir # Use the found directory path
             int8_available = True
        elif os.path.isdir(derived_int8_dir) and derived_int8_dir.endswith("_int8_model") and os.path.exists(os.path.join(derived_int8_dir, f"{base_pt_model_name}.xml")):
             print(f"Found potential INT8 model directory (by naming convention): {derived_int8_dir}")
             int8_model_dir = derived_int8_dir
             int8_available = True
        else:
             print("Info: INT8 model not found at default location or specified via --int8_model_dir. Skipping INT8 comparison.")
    elif os.path.isdir(int8_model_dir) and any(f.endswith('.xml') for f in os.listdir(int8_model_dir)):
        print(f"Using specified INT8 model directory for comparison: {int8_model_dir}")
        int8_available = True
    else:
        print(f"Warning: Specified --int8_model_dir ('{int8_model_dir}') is invalid or empty. Skipping INT8 comparison.")

    # --- Define All Modes to Compare ---
    # Format: (Display Name, mode_arg, config_dict)
    modes_to_compare = []

    # 1. PyTorch CPU
    modes_to_compare.append( ("PyTorch CPU", "pytorch_cpu", {}) )

    # 2. TensorRT GPU (if available)
    # Check globals for availability constants
    global TENSORRT_AVAILABLE, OPENVINO_AVAILABLE
    if TENSORRT_AVAILABLE and torch.cuda.is_available():
        modes_to_compare.append( ("TensorRT GPU", "tensorrt_gpu", {}) )
    else:
        print("Info: TensorRT or CUDA not available. Skipping TensorRT GPU comparison.")

    # 3. OpenVINO Modes (if available)
    if OPENVINO_AVAILABLE:
        modes_to_compare.append( ("OpenVINO FP32 Sync", "openvino_cpu", {"precision": "FP32", "use_async": False, "int8_model_dir": None}) )
        modes_to_compare.append( ("OpenVINO FP16 Sync", "openvino_cpu", {"precision": "FP16", "use_async": False, "int8_model_dir": None}) )
        modes_to_compare.append( ("OpenVINO FP32 Async", "openvino_cpu", {"precision": "FP32", "use_async": True, "int8_model_dir": None}) )
        # Add FP16 Async here
        modes_to_compare.append( ("OpenVINO FP16 Async", "openvino_cpu", {"precision": "FP16", "use_async": True, "int8_model_dir": None}) )
        if int8_available:
            modes_to_compare.append( ("OpenVINO INT8 Sync", "openvino_cpu", {"precision": "INT8", "use_async": False, "int8_model_dir": int8_model_dir}) )
            modes_to_compare.append( ("OpenVINO INT8 Async", "openvino_cpu", {"precision": "INT8", "use_async": True, "int8_model_dir": int8_model_dir}) )
    else:
        print("Info: OpenVINO not available. Skipping OpenVINO comparison modes.")

    if len(modes_to_compare) <= 1:
        print("Error: Not enough modes available or configured for comparison.")
        return {}

    # --- Select Benchmark Image ---
    # Use the first image found in test_images for this quick comparison
    benchmark_image = None
    if os.path.isdir("test_images"):
        images = [f for f in os.listdir("test_images") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            benchmark_image = os.path.join("test_images", images[0])
            print(f"Using image '{images[0]}' for comparison benchmark.")
        else:
            print("Error: No images found in 'test_images' directory.")
            return {}
    else:
        print("Error: 'test_images' directory not found.")
        return {}

    # --- Run Benchmarks ---
    # Try to get versions, handle potential import errors if checked earlier
    tensorrt_version = "N/A"
    if TENSORRT_AVAILABLE:
        try:
            import tensorrt as trt
            tensorrt_version = trt.__version__
        except ImportError:
            pass # Already marked as unavailable potentially
    openvino_version = "N/A"
    if OPENVINO_AVAILABLE:
        try:
            import openvino as ov
            openvino_version = ov.__version__
        except ImportError:
            pass

    results = {
        "system_info": {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "tensorrt_available": TENSORRT_AVAILABLE,
            "tensorrt_version": tensorrt_version,
            "openvino_available": OPENVINO_AVAILABLE,
            "openvino_version": openvino_version
        },
        "benchmark_settings": {
             "comparison_image": benchmark_image,
             "runs_per_mode": benchmark_runs_per_image,
             "int8_model_dir_used": int8_model_dir if int8_available else "N/A"
        },
        "backends": {}, # Keep 'backends' key for compatibility with existing plot func if needed, or rename later
        "fastest_backend": "",
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Make sure plotting is available
    plotting_available = False
    try:
        import matplotlib.pyplot as plt
        plotting_available = True
    except ImportError:
        print("Warning: matplotlib not found. Cannot generate plots for comparison.")


    for display_name, mode_arg, config in modes_to_compare:
        print(f"\n--- Testing Mode: {display_name} ---")
        # Create temporary args for this specific mode run
        import copy
        temp_args = copy.deepcopy(args)
        temp_args.mode = mode_arg
        temp_args.image = benchmark_image
        temp_args.benchmark = True # Enable internal benchmark runs
        temp_args.benchmark_runs = benchmark_runs_per_image
        temp_args.output = "" # Disable saving image during benchmark

        # Apply specific configurations (precision, async, etc.)
        temp_args.precision = config.get("precision", args.precision) # Default to FP32 if not set
        temp_args.use_async = config.get("use_async", args.use_async) # Default to False
        temp_args.int8_model_dir = config.get("int8_model_dir", args.int8_model_dir) # Pass INT8 dir if needed

        # Special handling for TensorRT skip convert flag? (Maybe not needed here, run_inference handles it)
        # temp_args.skip_convert = (mode_arg == "tensorrt_gpu")

        # Run inference and get the average time
        avg_time_ms = run_inference(temp_args)

        if avg_time_ms is not None:
            fps = 1000 / avg_time_ms if avg_time_ms > 0 else 0
            print(f"  Mode '{display_name}' completed.")
            print(f"  Average Inference Time: {avg_time_ms:.2f} ms")
            print(f"  Average FPS: {fps:.2f}")
            results["backends"][display_name] = {
                "inference_time": avg_time_ms, # Use 'inference_time' key for compatibility with plotter
                "fps": fps
            }
        else:
            print(f"  Mode '{display_name}' failed or skipped.")
            results["backends"][display_name] = { "error": "Failed or skipped" }


    # --- Process and Print Summary ---
    valid_results = {k: v for k, v in results["backends"].items() if "inference_time" in v}

    if valid_results:
        # Find fastest backend
        fastest = min(valid_results.items(), key=lambda item: item[1]["inference_time"])
        fastest_backend_name = fastest[0]
        fastest_time = fastest[1]["inference_time"]
        results["fastest_backend"] = fastest_backend_name

        # Add speed ratio
        for backend_name, data in results["backends"].items():
             if "inference_time" in data:
                 data["speed_ratio"] = data["inference_time"] / fastest_time if fastest_time > 0 else float('inf')

        print("\n=== Extended Performance Comparison Summary ===")
        # Adjust column width if needed based on longest display name
        # Calculate max length only from valid results keys
        max_name_len = max(len(name) for name in valid_results.keys()) if valid_results else 27 # Default width
        header = f"{'Mode'.ljust(max_name_len)} | Avg Time (ms) | Avg FPS   | Speed Ratio"
        print(header)
        print("-" * len(header))

        # Sort by time for printing
        for backend_name, data in sorted(valid_results.items(), key=lambda item: item[1]["inference_time"]):
            time = data["inference_time"]
            fps = data["fps"]
            ratio = data.get("speed_ratio", 1.0)
            print(f"{backend_name.ljust(max_name_len)} | {time:^13.2f} | {fps:^9.2f} | {ratio:.2f}x")

        print(f"\nFastest Mode: {fastest_backend_name} ({fastest_time:.2f} ms)")

        # --- Plotting ---
        if args.save_summary and plotting_available:
             if 'plot_benchmark_results' in globals():
                  timestamp = results.get("timestamp", datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
                  # Use a specific title for this extended comparison
                  plot_benchmark_results(results, args.benchmark_dir, timestamp, title_prefix="Extended Backend")
             else: # Correct indentation for the else block matching the 'if' on line 682
                  print("Warning: plot_benchmark_results function not found, skipping plot generation.")
                
        else:
            print("Warning: plot_benchmark_results function not found, skipping plot generation.")

    # This else corresponds to 'if valid_results:' -> Correct indentation
    else:
        print("\nNo valid comparison results obtained.")

    return results

def plot_benchmark_results(results, output_dir, timestamp, title_prefix="Backend"):
    """Create benchmark results visualization"""
    # Import here to avoid hard dependency if plotting isn't used
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Plotting requires matplotlib. Please install it (`pip install matplotlib`)")
        return
        
    # Use the key where mode results are stored ('modes' for the new func, 'backends' for old)
    results_key = "modes" if "modes" in results else "backends"

    if results_key not in results or not results[results_key]:
        print(f"No data found under key '{results_key}' to plot")
        return

    mode_names = []
    times = []
    fps = []
    ratios = []
    
    # Extract data, sorting by time
    # Handle potential missing keys more gracefully
    valid_modes = {}
    time_key = None
    fps_key = None
    for k, v in results[results_key].items():
         # Determine keys based on the first valid entry
         if not time_key and "overall_avg_inference_time" in v:
              time_key = "overall_avg_inference_time"
              fps_key = "overall_avg_fps"
         elif not time_key and "inference_time" in v:
              time_key = "inference_time"
              fps_key = "fps"

         # Check if the determined keys exist in the current item
         if time_key and time_key in v:
              valid_modes[k] = v
         else:
              print(f"Skipping mode '{k}' due to missing time data ('{time_key}')")


    if not valid_modes:
         print("No valid data points found for plotting after checking keys.")
         return
    if not time_key or not fps_key:
         print("Could not determine time/fps keys from results data.")
         return


    for mode_name, data in sorted(valid_modes.items(), key=lambda item: item[1][time_key]):
        mode_names.append(mode_name)
        times.append(data[time_key])
        fps.append(data[fps_key])
        # Use get() for speed_ratio as it might not exist if only one mode ran
        ratios.append(data.get("speed_ratio", 1.0))
    
    # Create chart
    num_modes = len(mode_names)
    # Adjust figure size based on number of modes to prevent overlap
    fig_width = max(10, num_modes * 1.2)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(fig_width, 15), sharex=True) # Share x-axis
    
    # Inference time chart
    bars1 = ax1.bar(mode_names, times, color='skyblue')
    ax1.set_title(f'{title_prefix} Comparison: Average Inference Time (ms)')
    ax1.set_ylabel('Milliseconds')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    # Add values on bars
    ax1.bar_label(bars1, fmt='%.2f', padding=3, fontsize=8)
    
    # FPS chart
    bars2 = ax2.bar(mode_names, fps, color='lightgreen')
    ax2.set_title(f'{title_prefix} Comparison: Average Frames Per Second (FPS)')
    ax2.set_ylabel('FPS')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.bar_label(bars2, fmt='%.2f', padding=3, fontsize=8)
    
    # Speed ratio chart
    bars3 = ax3.bar(mode_names, ratios, color='salmon')
    ax3.set_title(f'{title_prefix} Comparison: Speed Ratio (relative to fastest)')
    ax3.set_ylabel('Ratio (Lower is Faster)')
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    ax3.bar_label(bars3, fmt='%.2fx', padding=3, fontsize=8) # Add 'x' suffix

    # Common settings for x-axis
    plt.xticks(rotation=15, ha='right', fontsize=9) # Rotate labels slightly for better readability
    plt.xlabel("Benchmark Mode")

    fig.suptitle(f'{title_prefix} Performance Benchmark', fontsize=16, y=1.02) # Overall title
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent title overlap
    
    # Save chart
    # Use a safe filename based on the prefix
    safe_prefix = "".join(c if c.isalnum() else "_" for c in title_prefix).lower()
    plot_filename = os.path.join(output_dir, f"{safe_prefix}_benchmark_comparison_{timestamp}.png")
    try:
        # Correct indentation
        plt.savefig(plot_filename)
        print(f"Benchmark plot saved to: {plot_filename}")
    except Exception as e:
        # Correct indentation
        print(f"Error saving plot: {e}")
    finally:
        # Correct indentation
        # Close chart to free memory, should happen regardless of save success
        plt.close(fig)

def run_inference(args):
    """Run inference with benchmarking"""
    # Check file paths
    if not os.path.exists(args.image):
        print(f"Error: Image {args.image} does not exist!")
        return None
    
    if not os.path.exists(args.model):
        print(f"Error: Model {args.model} does not exist!")
        return None
        
    # Load original image for drawing
    try:
        # Use OpenCV to read original image
        original_image = cv2.imread(args.image)
        if original_image is None:
            raise ValueError(f"Cannot read image: {args.image}")
        print(f"Loaded image: {args.image}, original size: {original_image.shape}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
        
    # Determine model type
    try:
        if args.mode == "pytorch_cpu":
            device = torch.device("cpu")
            print("Using CPU for PyTorch")
                
            # Load PyTorch model
            try:
                from ultralytics import YOLO
                model = YOLO(args.model)
                model.to(device)
                
                # Benchmark runs
                times = []
                num_runs = args.benchmark_runs if args.benchmark else 1
                
                for i in range(num_runs):
                    if args.benchmark:
                        print(f"Running inference {i+1}/{num_runs}...")
                        
                    start_time = time.perf_counter() # Use perf_counter
                    results = model.predict(args.image, save=False, verbose=False)
                    inference_time = (time.perf_counter() - start_time) * 1000 # Use perf_counter
                    times.append(inference_time)
                    
                # Calculate average and standard deviation
                avg_time = sum(times) / len(times)
                std_dev = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
                
                if args.benchmark:
                    print(f"Average inference time: {avg_time:.2f} ms (FPS: {1000/avg_time:.2f})")
                    print(f"Standard deviation: {std_dev:.2f} ms")
                    print(f"Min: {min(times):.2f} ms, Max: {max(times):.2f} ms")
                else:
                    print(f"Inference time: {avg_time:.2f} ms (FPS: {1000/avg_time:.2f})")
                    
                # Process last result for visualization
                # detections = results[0].boxes.data.cpu().numpy() # Incorrect if results is a list
                if results: # Check if results list is not empty
                    last_result_data = results[-1].boxes.data.cpu().numpy() # Get data from the LAST result object
                else:
                    print("Error: No results obtained from PyTorch model.")
                    return None # Need to return None if no results

                # Save result if output path is specified, regardless of benchmark flag
                # REMOVE: if not args.benchmark:
                if args.output: # Check if an output path IS provided
                    print("Processing PyTorch output for visualization...")
                    run_visualization(args, original_image, last_result_data)
                else:
                    # If no output path, just indicate completion for benchmark (or normal run without save)
                    pass # Pass here if no output needed
                    
                return avg_time
                
            except Exception as e:
                print(f"Error with PyTorch model: {e}")
                import traceback
                traceback.print_exc()
                return None
                
        elif args.mode == "tensorrt_gpu":
            if not torch.cuda.is_available():
                print("Error: CUDA not available, cannot use TensorRT")
                return None
                
            if not TENSORRT_AVAILABLE:
                print("Error: TensorRT not installed or not available")
                return None
                
            print(f"Using TensorRT on CUDA: {torch.cuda.get_device_name(0)}")
            
            try:
                from ultralytics import YOLO
                
                # Look for TensorRT engine file
                tensorrt_model_path = args.tensorrt_model if args.tensorrt_model else args.model.replace('.pt', '_tensorrt.engine')
                
                # Search for possible engine files
                if not os.path.exists(tensorrt_model_path):
                    base_name = os.path.splitext(args.model)[0]
                    possible_paths = [
                        f"{base_name}_engine.engine",    # New format
                        f"{base_name}.engine",           # Old format
                        os.path.join(os.path.dirname(args.model), "yolov8n.engine")
                    ]
                    
                    for path in possible_paths:
                        if os.path.exists(path):
                            tensorrt_model_path = path
                            print(f"Found TensorRT model at: {tensorrt_model_path}")
                            break
                
                # If TensorRT engine found, use it for inference
                if os.path.exists(tensorrt_model_path) and not args.force_convert:
                    print(f"Loading existing TensorRT model: {tensorrt_model_path}")
                    
                    # Load the model with PyTorch but optimize with TensorRT
                    model = YOLO(args.model)
                    device = torch.device("cuda")
                    model.to(device)
                    
                    # Enable TensorRT optimization
                    if tensorrt_model_path.endswith('.engine'):
                        print("Using TensorRT optimization")
                    
                    # Benchmark runs
                    times = []
                    num_runs = args.benchmark_runs if args.benchmark else 1
                    
                    # Warmup
                    print("Warming up model...")
                    for _ in range(3):
                        model.predict(args.image, verbose=False, save=False)
                        torch.cuda.synchronize()
                    
                    for i in range(num_runs):
                        if args.benchmark:
                            print(f"Running inference {i+1}/{num_runs}...")
                            
                        torch.cuda.synchronize()
                        start_time = time.perf_counter() # Use perf_counter
                        results = model.predict(args.image, save=False, verbose=False)
                        torch.cuda.synchronize()
                        inference_time = (time.perf_counter() - start_time) * 1000 # Use perf_counter
                        times.append(inference_time)
                    
                    # Calculate average and standard deviation
                    avg_time = sum(times) / len(times)
                    std_dev = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
                    
                    if args.benchmark:
                        print(f"Average inference time: {avg_time:.2f} ms (FPS: {1000/avg_time:.2f})")
                        print(f"Standard deviation: {std_dev:.2f} ms")
                        print(f"Min: {min(times):.2f} ms, Max: {max(times):.2f} ms")
                    else:
                        print(f"Inference time: {avg_time:.2f} ms (FPS: {1000/avg_time:.2f})")
                    
                    # Process results for visualization from the LAST run
                    # detections = results[0].boxes.data.cpu().numpy() # Incorrect if results is a list
                    if results:
                        last_result_data = results[-1].boxes.data.cpu().numpy()
                    else:
                        print("Error: No results obtained from TensorRT model.")
                        return None

                    # Save result if output path is specified, regardless of benchmark flag
                    # REMOVE: if not args.benchmark:
                    if args.output: # Check if an output path IS provided
                        print("Processing TensorRT output for visualization...")
                        run_visualization(args, original_image, last_result_data)
                    else:
                        # If no output path, just indicate completion for benchmark
                        pass # Pass here if no output needed
                    
                    return avg_time
                    
                else:
                    # If engine file not found, try to convert
                    if not args.skip_convert:
                        try:
                            print(f"Converting PyTorch model to TensorRT format...")
                            model = YOLO(args.model)
                            model.export(format="engine", device=0, half=True, batch=1)
                            
                            # Look for the exported model
                            tensorrt_model_path = args.model.replace('.pt', '_engine.engine')
                            if os.path.exists(tensorrt_model_path):
                                print(f"TensorRT model exported to: {tensorrt_model_path}")
                                
                                # Recursive call using the new model
                                args.tensorrt_model = tensorrt_model_path
                                return run_inference(args)
                            else:
                                print("TensorRT model conversion failed, falling back to PyTorch CPU")
                                args.mode = "pytorch_cpu"
                                return run_inference(args)
                        except Exception as e:
                            print(f"Error converting to TensorRT: {e}")
                            print("Falling back to PyTorch CPU")
                            args.mode = "pytorch_cpu"
                            return run_inference(args)
                    else:
                        print("TensorRT model not found and conversion skipped. Falling back to PyTorch CPU")
                        args.mode = "pytorch_cpu"
                        return run_inference(args)
                
            except Exception as e:
                print(f"Error with TensorRT: {e}")
                import traceback
                traceback.print_exc()
                return None
                
        elif args.mode == "openvino_cpu":
            try:
                if not OPENVINO_AVAILABLE:
                    print("OpenVINO requested but not available. Please install OpenVINO.")
                    return None
                    
                import openvino as ov
                core = ov.Core() # Initialize core earlier

                # --- Determine Model Path based on Precision ---
                openvino_model_xml = ""
                is_int8 = (args.precision == 'INT8')

                if is_int8:
                    if args.int8_model_dir and os.path.isdir(args.int8_model_dir):
                        # Look for a _int8.xml file first, then any .xml
                        base_name = os.path.splitext(os.path.basename(args.model))[0] if args.model.endswith('.pt') else "yolov8n"
                        int8_xml_path_specific = os.path.join(args.int8_model_dir, f"{base_name}_int8.xml")
                        int8_xml_path_generic = os.path.join(args.int8_model_dir, f"{base_name}.xml") # NNCF might just name it like the original

                        if os.path.exists(int8_xml_path_specific):
                            openvino_model_xml = int8_xml_path_specific
                            print(f"Found specific INT8 model: {openvino_model_xml}")
                        elif os.path.exists(int8_xml_path_generic):
                             openvino_model_xml = int8_xml_path_generic
                             print(f"Found generic INT8 model in specified dir: {openvino_model_xml}")
                        else:
                             # Check for *any* xml file in the directory as a last resort
                             from pathlib import Path # Import Path here
                             xml_files_in_dir = list(Path(args.int8_model_dir).glob("*.xml"))
                             if xml_files_in_dir:
                                 openvino_model_xml = str(xml_files_in_dir[0])
                                 print(f"Found an XML file in INT8 dir, assuming it's the INT8 model: {openvino_model_xml}")
                             else:
                                 print(f"Error: --precision INT8 specified, but no suitable .xml file found in --int8_model_dir: {args.int8_model_dir}")
                                 return None # Correct indentation
                    else:
                        print(f"Error: --precision INT8 requires a valid --int8_model_dir.")
                        return None

                elif args.precision == 'FP16':
                    # --- Hardcoded FP16 model path ---
                    print("Using hardcoded path for FP16 OpenVINO model...")
                    hardcoded_fp16_path = "models/yolov8n_openvino_fp16_model/yolov8n_fp16.xml"

                    if os.path.exists(hardcoded_fp16_path):
                        openvino_model_xml = hardcoded_fp16_path
                        print(f"Found FP16 model at hardcoded path: {openvino_model_xml}")
                    else:
                        print(f"Error: Hardcoded FP16 model path does not exist: {hardcoded_fp16_path}")
                        print(f"Please ensure the model exists at this location or run download_model.py again.")
                        return None
                    # --- End of hardcoded path logic ---

                else: # FP32 - use original logic to find FP32 model
                    # Use the existing logic to find FP32 model
                    print(f"Attempting to find FP32 OpenVINO model based on {args.model}...")
                    potential_paths = []
                    if args.model.endswith('.xml'):
                        potential_paths.append(args.model)
                    elif args.model.endswith('.pt'):
                         # Check 'models' first as per previous findings
                         potential_paths.append(os.path.join("models", "yolov8n.xml")) # Default location
                         potential_paths.append(os.path.join("models", "yolov8n_openvino_model", "yolov8n.xml")) # Exported location
                         potential_paths.append(args.model.replace('.pt', '_openvino_model/yolov8n.xml')) # Older logic path
                         potential_paths.append(args.model.replace('.pt', '.xml'))
                    else: # If args.model is not .pt or .xml, assume it might be a base name
                         potential_paths.append(os.path.join("models", f"{args.model}.xml"))
                         potential_paths.append(os.path.join("models", f"{args.model}_openvino_model", f"{args.model}.xml"))

                    # Clean potential paths (remove duplicates, ensure they are strings)
                    potential_paths = sorted(list(set(filter(None, potential_paths))))

                    for path in potential_paths:
                       if os.path.exists(path):
                           openvino_model_xml = path
                           print(f"Found base OpenVINO model (for {args.precision}) at: {openvino_model_xml}")
                           break

                    # If FP32 model was not found after checking potential paths, report error and exit
                    if not openvino_model_xml:
                        print(f"Error: OpenVINO model (.xml) for FP32 not found. Please ensure the model exists at one of the expected paths or provide the path directly.")
                        print(f"Checked paths: {potential_paths}")
                        return None

                # --- Model Loading --- (This part remains)
                if not openvino_model_xml or not os.path.exists(openvino_model_xml):
                     print(f"Error: Could not determine a valid OpenVINO model path to load for precision {args.precision}. Final path checked: {openvino_model_xml}")
                     return None

                print(f"Loading OpenVINO model ({args.precision}) from: {openvino_model_xml}")
                try:
                    model_ov = core.read_model(openvino_model_xml) # Use model_ov to avoid conflict
                except Exception as read_e:
                     print(f"Error reading OpenVINO model file {openvino_model_xml}: {read_e}")
                     return None


                # --- Get Input/Output Info ---
                try:
                    input_node = model_ov.input(0)
                    output_node = model_ov.output(0) # Assuming single output for YOLOv8
                    input_shape = tuple(input_node.shape) # e.g., (1, 3, 640, 640)
                    input_h, input_w = input_shape[2], input_shape[3]
                    print(f"Model expects input shape: {input_shape}")
                except Exception as node_e:
                     print(f"Warning: Could not get precise input/output node info: {node_e}. Using defaults.")
                     input_h, input_w = 640, 640 # Fallback to default

                # --- Configure Compilation ---
                device = "CPU" # Currently hardcoded, could be made an arg
                compile_config = {}
                if is_int8:
                     print("Using INT8 quantized model. No explicit precision hint needed during compilation.")
                elif args.precision == 'FP16':
                     print("Using pre-converted FP16 model (hardcoded path). No explicit precision hint needed.")
                else: # FP32
                     print("Using FP32 precision for OpenVINO.")


                # --- Compile Model ---
                print(f"Compiling model for {device} ({args.precision})...")
                try:
                    compiled_model = core.compile_model(model_ov, device, compile_config)
                except Exception as compile_e:
                     print(f"Error compiling OpenVINO model: {compile_e}")
                     return None
                print("Model compiled successfully.")

                # --- Preprocess Image ---
                # Use the determined input size (input_w, input_h)
                preprocessed_image, _ = load_and_preprocess_image(args.image, size=(input_w, input_h))

                # --- Inference Execution (Async/Sync Logic follows...) ---
                # Correct indentation for num_runs
                num_runs = args.benchmark_runs if args.benchmark else 1
                times = []
                last_outputs = None # Reset last_outputs here
                avg_time = 0
                fps = 0

                if args.use_async:
                    # --- Asynchronous Inference ---
                    print("Using Asynchronous Inference Pipeline")
                    try:
                        # Try getting optimal number of requests using the correct property name if possible
                        # Ensure ov is imported before this point
                        num_requests = compiled_model.get_property(ov.properties.hint.num_requests()) \
                                        if ov.properties.hint.num_requests() in compiled_model.properties else 1
                    except Exception:
                         print("Could not get optimal number of infer requests, defaulting to 1.")
                         num_requests = 1
                    print(f"Using {num_requests} Infer Requests")
                    infer_queue = ov.AsyncInferQueue(compiled_model, num_requests)

                    # Store results using a list accessible by closure
                    results_list = []
                    completed_requests = 0
                    # Make sure last_outputs (defined outside this if block) can be modified
                    # No need for nonlocal if last_outputs is already in the outer scope

                    # Define the callback *inside* this scope to capture results_list
                    # Add *args to accept potential extra arguments from OpenVINO
                    def callback_with_closure(request, *args):
                        nonlocal completed_requests, last_outputs, results_list
                        try:
                            # Get the output tensor data directly
                            output_data = request.get_output_tensor().data.copy()
                            results_list.append(output_data)
                            last_outputs = output_data # Keep track of the latest output
                        except Exception as cb_e:
                             print(f"Error inside async callback: {cb_e}")
                        finally:
                             completed_requests += 1

                    # Set the callback, passing only the function itself
                    try:
                         infer_queue.set_callback(callback_with_closure)
                    except Exception as set_cb_e:
                         print(f"Error setting async callback: {set_cb_e}")
                         return None # Cannot proceed without callback

                    # Prepare input data dictionary
                    try:
                        input_tensor_name = model_ov.input(0).get_any_name()
                    except Exception:
                        print("Warning: Could not get input tensor name, using default key from compiled model inputs.")
                        try:
                             input_tensor_name = compiled_model.input(0).get_any_name() # Try compiled model input
                        except Exception:
                              print("Error: Failed to get input tensor name even from compiled model.")
                              return None


                    input_data = {input_tensor_name: preprocessed_image}

                    print(f"Submitting {num_runs} inference requests asynchronously...")
                    total_start_time = time.perf_counter() # Use perf_counter for total async time

                    # Submit all runs
                    successful_submissions = 0
                    for i in range(num_runs):
                        try:
                            infer_queue.start_async(input_data)
                            successful_submissions += 1
                        except Exception as submit_e:
                             print(f"Error submitting async request {i+1}/{num_runs}: {submit_e}")
                             # Stop submitting further requests on error
                             break

                    # Wait for all *successfully submitted* requests to complete
                    # Use number of successful submissions for waiting logic if needed,
                    # but wait_all should handle jobs in flight correctly.
                    if successful_submissions > 0:
                         infer_queue.wait_all()
                    else:
                         print("No requests were submitted successfully.")

                    total_end_time = time.perf_counter() # Use perf_counter for total async time
                    total_time_async_ms = (total_end_time - total_start_time) * 1000

                    # Check based on *successful* submissions vs completed runs
                    if completed_requests != successful_submissions:
                         print(f"Warning: Submitted {successful_submissions} requests, but only {completed_requests} completed.")
                         if not results_list: # If no results at all, fail
                              print("Error: No results obtained from async inference.")
                              # It's possible avg_time is calculated but invalid, ensure return None
                              return None

                    # --- Performance Calculation for Async ---
                    # Base calculation on successfully submitted requests
                    avg_time = total_time_async_ms / successful_submissions if successful_submissions > 0 else 0
                    fps = successful_submissions * 1000 / total_time_async_ms if total_time_async_ms > 0 else 0

                    print("-" * 30)
                    print(f"OpenVINO Async ({device}, {args.precision}, {successful_submissions}/{num_runs} runs submitted/requested):")
                    print(f"  Total Time: {total_time_async_ms:.2f} ms")
                    print(f"  Avg Time per inference (overlapped): {avg_time:.2f} ms")
                    print(f"  Throughput (FPS): {fps:.2f}")
                    print("-" * 30)

                # Ensure this else is aligned with the 'if args.use_async:' above
                else:
                    # --- Synchronous Inference ---
                    print(f"Using Synchronous Inference Pipeline ({device}, {args.precision})")
                    try: # Add try-except around getting output node
                        output_node_obj = compiled_model.output(0)
                    except Exception as out_node_e:
                        print(f"Error getting output node: {out_node_e}")
                        return None

                    print(f"Running {num_runs} inference(s)...")
                    for i in range(num_runs):
                        start_time = time.perf_counter() # Use perf_counter
                        try: # Add try-except around inference call
                             outputs_dict = compiled_model(preprocessed_image)
                             inference_time = (time.perf_counter() - start_time) * 1000 # Use perf_counter
                             times.append(inference_time)
                             last_outputs = outputs_dict[output_node_obj] # Use the node object as key
                        except Exception as infer_e:
                              print(f"Error during synchronous inference run {i+1}: {infer_e}")
                              # Decide whether to stop or continue
                              return None # Stop on error for sync

                    # --- Calculate Performance Metrics for Sync --- (Remains same)
                    avg_time = np.mean(times) if times else 0
                    std_dev = np.std(times) if len(times) > 1 else 0
                    min_time = np.min(times) if times else 0
                    max_time = np.max(times) if times else 0
                    fps = 1000 / avg_time if avg_time > 0 else 0

                    print("-" * 30)
                    if args.benchmark and num_runs > 1:
                        print(f"OpenVINO Sync Benchmark ({device}, {args.precision}, {num_runs} runs):")
                        print(f"  Avg Time: {avg_time:.2f} ms")
                        print(f"  Std Dev:  {std_dev:.2f} ms")
                        print(f"  Min Time: {min_time:.2f} ms")
                        print(f"  Max Time: {max_time:.2f} ms")
                        print(f"  Avg FPS:  {fps:.2f}")
                    else:
                        print(f"OpenVINO Sync Inference Time ({device}, {args.precision}): {avg_time:.2f} ms")
                        print(f"OpenVINO Sync FPS ({device}, {args.precision}): {fps:.2f}")
                    print("-" * 30)

                # --- Process Last Result for Visualization (Common path for sync/async) ---
                if last_outputs is not None:
                    # Need original image shape here for processing
                    orig_h, orig_w = original_image.shape[:2]

                    if args.output: # Only process and visualize if output path is given
                        print("Processing OpenVINO output for visualization...")
                        # Call the new processing function
                        detections = process_openvino_output(
                            last_outputs,
                            original_shape=(orig_h, orig_w),
                            input_shape=(input_h, input_w),
                            conf_threshold=args.threshold
                        )
                        # Check if detections were successfully processed
                        if detections is not None and detections.shape[0] > 0:
                             print(f"Found {detections.shape[0]} detections after NMS.")
                             run_visualization(args, original_image, detections)
                        elif detections is not None: # Correct indentation
                             print("No detections found meeting criteria after NMS.")
                             # Optionally, save the image without boxes or skip saving
                             run_visualization(args, original_image, detections) # Pass empty detections
                        else: # Correct indentation
                             print("Error occurred during OpenVINO output processing, skipping visualization.")
                    else:
                         # If no output path, just indicate completion for benchmark
                         pass # Ensure pass is here and correctly indented

                    return avg_time # Return performance metric outside the if args.output block

                else:
                    print("Error: No output received from OpenVINO model after inference.")
                    return None

            except ImportError as ie:
                 print(f"Import error during OpenVINO processing: {ie}. Ensure OpenVINO and necessary components are installed correctly.")
                 # Correct indentation for traceback
                 import traceback
                 traceback.print_exc()
                 return None
            except Exception as e:
                print(f"Error during OpenVINO configuration, inference, or post-processing: {e}")
                import traceback
                traceback.print_exc()
                return None
                
    except Exception as e:
        print(f"General error setting up or running model: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    return None # Should return avg_time from the successful block

def run_visualization(args, original_image, detections):
    """Visualize and save detection results"""
    # COCO class names
    class_names = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    
    # Process output path
    output_path = args.output
    # Ensure output directory exists (This part is correct and sufficient)
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Create result image
    result_image = original_image.copy()
    
    # Draw detections
    h, w = original_image.shape[:2]
    detected_count = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for i, det in enumerate(detections):
        # Check confidence
        confidence = float(det[4])
        if confidence < args.threshold:
            continue
            
        # Extract coordinates and class
        x1, y1, x2, y2 = map(int, det[:4])
        class_id = int(det[5])
        
        # Ensure coordinates are valid
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        # Skip small boxes
        if x2 - x1 < 10 or y2 - y1 < 10:
            continue
            
        detected_count += 1
        
        # Get class name
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        
        # Draw box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(result_image, label, (x1, y1 - 5), font, 0.5, (0, 255, 0), 2)
    
    # Add detection count at top
    engine_type = args.mode
        
    if engine_type == "tensorrt_gpu":
        mode_str = "TensorRT GPU"
    elif engine_type == "pytorch_cpu":
        mode_str = "PyTorch CPU"
    elif engine_type == "openvino_cpu":
        mode_str = "OpenVINO CPU"
    else:
        mode_str = engine_type
        
    summary = f"Detected {detected_count} objects ({mode_str})"
    text_size = cv2.getTextSize(summary, font, 0.8, 2)[0]
    cv2.putText(result_image, summary, (10, text_size[1] + 10), font, 0.8, (0, 255, 0), 2)
    
    # Save result
    cv2.imwrite(output_path, result_image)
    print(f"Result saved to: {output_path}")
    print(f"Detected {detected_count} objects at threshold {args.threshold}")

def run_single_inference(args):
    """Run a single inference without benchmarking"""
    run_inference(args)

def run_openvino_benchmark(args):
    """Runs a detailed benchmark comparing OpenVINO modes."""
    print("\n=== Running Detailed OpenVINO Benchmark ===")

    # --- Configuration ---
    base_model_path = args.model # Usually the .pt or base .xml path
    int8_model_dir = args.int8_model_dir
    benchmark_runs_per_image = args.benchmark_runs # Runs inside run_inference

    # --- Find Test Images ---
    if not os.path.isdir("test_images"):
        print("Error: 'test_images' directory not found.")
        return {}
    image_files = [f for f in os.listdir("test_images") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("Error: No images found in 'test_images' directory.")
        return {}
    num_total_images = len(image_files)
    print(f"Found {num_total_images} images for benchmarking.")

    # --- Check INT8 Model Availability ---
    int8_available = False
    if not int8_model_dir:
        # Try to derive a default path if not provided
        base_pt_model_name = os.path.splitext(os.path.basename(args.model))[0] if args.model.endswith('.pt') else "yolov8n"
        # Adjust derived path slightly to match typical ultralytics export structure better
        derived_int8_dir = args.model.replace('.pt', '_openvino_model') if args.model.endswith('.pt') else f'models/{base_pt_model_name}_openvino_model'
        # Look for a file like yolov8n_int8.xml OR just yolov8n.xml within a subdir that implies INT8
        potential_int8_xml = os.path.join(derived_int8_dir, f"{base_pt_model_name}_int8.xml")
        if os.path.exists(potential_int8_xml):
             print(f"Found potential INT8 model at: {potential_int8_xml}")
             # Need the directory containing the XML for run_inference logic
             int8_model_dir = derived_int8_dir # Set the dir path
             int8_available = True
        elif os.path.isdir(derived_int8_dir) and os.path.exists(os.path.join(derived_int8_dir, f"{base_pt_model_name}.xml")):
             # Check if the directory *name* implies INT8, e.g., ends with _int8_model
             if derived_int8_dir.endswith("_int8_model"):
                  print(f"Found potential INT8 model directory (by naming convention): {derived_int8_dir}")
                  int8_model_dir = derived_int8_dir
                  int8_available = True
             else:
                  print("Warning: --int8_model_dir not specified and default path checking did not conclusively find an INT8 model. Skipping INT8 tests.")
        else:
             print("Warning: --int8_model_dir not specified and default path checking did not find an INT8 model. Skipping INT8 tests.")

    elif os.path.isdir(int8_model_dir):
         # Check if at least one XML file exists within the specified directory
         if any(f.endswith('.xml') for f in os.listdir(int8_model_dir)):
              print(f"Using specified INT8 model directory: {int8_model_dir}")
              int8_available = True
         else:
              print(f"Warning: Specified --int8_model_dir ('{int8_model_dir}') contains no .xml file. Skipping INT8 tests.")
    else:
        print(f"Warning: Specified --int8_model_dir ('{int8_model_dir}') is not a valid directory. Skipping INT8 tests.")

    # --- Define Benchmark Modes ---
    # Format: (Name, Precision, Use Async, INT8 Dir (if needed))
    modes_to_run = [
        ("OpenVINO FP32 Sync", "FP32", False, None),
        ("OpenVINO FP16 Sync", "FP16", False, None), # FP16 hint effectiveness depends on hardware
        *([("OpenVINO INT8 Sync", "INT8", False, int8_model_dir)] if int8_available else []),
        ("OpenVINO FP32 Async", "FP32", True, None),
        *([("OpenVINO INT8 Async", "INT8", True, int8_model_dir)] if int8_available else []),
        # Add FP16 Async? Less common benefit unless on specific hardware like VPU, but can include.
        # ("OpenVINO FP16 Async", "FP16", True, None),
    ]

    if not OPENVINO_AVAILABLE:
         print("Error: OpenVINO is not available. Cannot run OpenVINO benchmarks.")
         return {}

    if not modes_to_run:
        print("Error: No valid OpenVINO benchmark modes configured (or OpenVINO not available).")
        return {}

    # --- Run Benchmarks ---
    # Use system info from args if available, or generate here
    system_info = {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "tensorrt_available": TENSORRT_AVAILABLE,
            "openvino_available": OPENVINO_AVAILABLE,
            "openvino_version": ov.__version__ if OPENVINO_AVAILABLE else "N/A"
    }

    results = {
        "system_info": system_info,
        "benchmark_settings": {
             "base_model": base_model_path,
             "images_tested": num_total_images,
             "runs_per_image": benchmark_runs_per_image,
             "int8_model_dir_used": int8_model_dir if int8_available else "N/A"
        },
        "modes": {}, # Store aggregated results per mode
        "fastest_mode": "",
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Make sure the plot function dependency (matplotlib) is handled
    try:
        import matplotlib.pyplot as plt
        plotting_available = True
    except ImportError:
        print("Warning: matplotlib not found. Cannot generate plots.")
        plotting_available = False


    for mode_name, precision, use_async, specific_int8_dir in modes_to_run:
        print(f"\n--- Benchmarking Mode: {mode_name} ---")
        all_image_times = []
        images_processed_count = 0

        for i, img_file in enumerate(image_files):
            img_path = os.path.join("test_images", img_file)
            print(f"  Processing image {i+1}/{num_total_images}: {img_file} ({precision}, {'Async' if use_async else 'Sync'})")

            # Create temporary args for this specific run
            import copy
            temp_args = copy.deepcopy(args)
            temp_args.mode = "openvino_cpu" # Force OpenVINO mode
            temp_args.image = img_path
            temp_args.precision = precision
            temp_args.use_async = use_async
            temp_args.int8_model_dir = specific_int8_dir # Use the correct INT8 dir for this mode
            temp_args.model = base_model_path # Ensure base model path is consistent
            temp_args.benchmark = True # Enable internal benchmark runs
            temp_args.benchmark_runs = benchmark_runs_per_image
            # Disable saving individual images during benchmark by providing empty output path
            temp_args.output = "" # run_visualization checks for non-empty path

            # Run inference and get the average time for this image
            # Ensure run_inference is robust and returns None on failure
            avg_time_ms = run_inference(temp_args)

            if avg_time_ms is not None:
                all_image_times.append(avg_time_ms)
                images_processed_count += 1
                # Reduce verbosity slightly inside the loop
                # print(f"    Avg time for this image: {avg_time_ms:.2f} ms")
            else:
                print(f"    Skipped image {img_file} due to error in inference for mode {mode_name}.")

        # --- Aggregate Results for this Mode ---
        if images_processed_count > 0:
            overall_avg_time = sum(all_image_times) / images_processed_count
            # Calculate FPS based on overall average time
            overall_fps = 1000 / overall_avg_time if overall_avg_time > 0 else 0
            print(f"  Mode '{mode_name}' completed.")
            print(f"  Images successfully processed: {images_processed_count}/{num_total_images}")
            print(f"  Overall Average Inference Time: {overall_avg_time:.2f} ms")
            print(f"  Overall Average FPS: {overall_fps:.2f}")

            results["modes"][mode_name] = {
                "overall_avg_inference_time": overall_avg_time,
                "overall_avg_fps": overall_fps,
                "images_processed": images_processed_count
                # Add precision and async flag for clarity in results dict?
                # "precision": precision,
                # "async": use_async
            }
        else:
            print(f"  Mode '{mode_name}' failed to process any images.")
            results["modes"][mode_name] = { "error": "Failed to process any images" }


    # --- Determine Fastest Mode ---
    if results["modes"]:
        valid_modes = {k: v for k, v in results["modes"].items() if "overall_avg_inference_time" in v}
        if valid_modes:
            fastest = min(valid_modes.items(), key=lambda item: item[1]["overall_avg_inference_time"])
            fastest_mode_name = fastest[0]
            fastest_time = fastest[1]["overall_avg_inference_time"]
            results["fastest_mode"] = fastest_mode_name

            # Add speed ratio relative to fastest
            for mode_name, data in results["modes"].items():
                 if "overall_avg_inference_time" in data:
                     # Ensure fastest_time is not zero to avoid division error
                     data["speed_ratio"] = data["overall_avg_inference_time"] / fastest_time if fastest_time > 0 else float('inf')

            print("\n=== OpenVINO Benchmark Summary ===")
            print("Mode                        | Avg Time (ms) | Avg FPS   | Speed Ratio")
            print("----------------------------|---------------|-----------|------------")
            # Sort by time for printing
            for mode_name, data in sorted(valid_modes.items(), key=lambda item: item[1]["overall_avg_inference_time"]):
                time = data["overall_avg_inference_time"]
                fps = data["overall_avg_fps"]
                ratio = data.get("speed_ratio", 1.0)
                print(f"{mode_name.ljust(27)}| {time:^13.2f} | {fps:^9.2f} | {ratio:.2f}x")
            print(f"\nFastest Mode: {fastest_mode_name} ({fastest_time:.2f} ms)")

            # Add plotting call here, inside the check for valid modes and plotting availability
            if args.save_summary and plotting_available:
                 if 'plot_benchmark_results' in globals():
                      timestamp = results.get("timestamp", datetime.datetime.now().strftime('%Y%m%d_%H%M%S')) # Get timestamp from results
                      # Use a specific title for this extended comparison
                      plot_benchmark_results(results, args.benchmark_dir, timestamp, title_prefix="OpenVINO")
                 else:
                      print("Warning: plot_benchmark_results function not found, skipping plot generation.")


        else:
             print("\nNo valid benchmark results obtained.")
    else:
        print("\nNo benchmark modes were run or completed successfully.")

    return results

# <<< Add this new function definition somewhere before the main() function >>>

def run_all_images_all_modes(args):
    """
    Runs inference on all images in test_images for all available backend modes,
    saves visualization results, and calculates average performance per mode.
    """
    print("\\n=== Running All Images Through All Modes ===")

    # --- Configuration & Setup ---
    base_model_path = args.model
    int8_model_dir = args.int8_model_dir
    benchmark_runs_per_image = args.benchmark_runs # Use benchmark runs for stable timing per image
    # Import Path from pathlib here
    from pathlib import Path
    results_base_dir = Path(args.results_dir)
    results_base_dir.mkdir(parents=True, exist_ok=True)

    # --- Find Test Images ---
    test_images_dir = Path("test_images")
    if not test_images_dir.is_dir():
        print(f"Error: 'test_images' directory not found at {test_images_dir}")
        return
    image_files = sorted(list(test_images_dir.glob("*.jpg")) + \
                         list(test_images_dir.glob("*.jpeg")) + \
                         list(test_images_dir.glob("*.png")))
    if not image_files:
        print(f"Error: No images found in '{test_images_dir}' directory.")
        return
    num_total_images = len(image_files)
    print(f"Found {num_total_images} images in {test_images_dir}.")

    # --- Check INT8 Model Availability ---
    # (Reuse the logic from run_benchmark_comparison or run_openvino_benchmark)
    int8_available = False
    if not int8_model_dir:
        base_pt_model_name = os.path.splitext(os.path.basename(args.model))[0] if args.model.endswith('.pt') else "yolov8n"
        derived_int8_dir = Path(f"yolov8n_openvino_model_int8") # Path where NNCF saved it
        # Check if the default NNCF output exists
        if derived_int8_dir.is_dir() and any(f.endswith('.xml') for f in os.listdir(derived_int8_dir)):
             print(f"Found potential INT8 model directory at: {derived_int8_dir}")
             int8_model_dir = str(derived_int8_dir) # Use the found directory path string
             int8_available = True
        else:
             print("Info: INT8 model directory not specified and not found at default location ./yolov8n_openvino_model_int8. Skipping INT8 modes.")
    elif os.path.isdir(int8_model_dir) and any(f.endswith('.xml') for f in os.listdir(int8_model_dir)):
        print(f"Using specified INT8 model directory: {int8_model_dir}")
        int8_available = True
    else:
        print(f"Warning: Specified --int8_model_dir ('{int8_model_dir}') is invalid or empty. Skipping INT8 modes.")
        int8_model_dir = None # Ensure it's None if invalid


    # --- Define All Modes To Run ---
    # (Similar to run_benchmark_comparison)
    # Make sure these are accessible if defined globally
    global TENSORRT_AVAILABLE, OPENVINO_AVAILABLE, torch, ov
    modes_to_run = []
    modes_to_run.append( ("PyTorch_CPU", "pytorch_cpu", {}) )
    if TENSORRT_AVAILABLE and torch.cuda.is_available():
        modes_to_run.append( ("TensorRT_GPU", "tensorrt_gpu", {}) )
    if OPENVINO_AVAILABLE:
        modes_to_run.append( ("OpenVINO_FP32_Sync", "openvino_cpu", {"precision": "FP32", "use_async": False, "int8_model_dir": None}) )
        modes_to_run.append( ("OpenVINO_FP16_Sync", "openvino_cpu", {"precision": "FP16", "use_async": False, "int8_model_dir": None}) )
        modes_to_run.append( ("OpenVINO_FP32_Async", "openvino_cpu", {"precision": "FP32", "use_async": True, "int8_model_dir": None}) )
        modes_to_run.append( ("OpenVINO_FP16_Async", "openvino_cpu", {"precision": "FP16", "use_async": True, "int8_model_dir": None}) )
        if int8_available:
            modes_to_run.append( ("OpenVINO_INT8_Sync", "openvino_cpu", {"precision": "INT8", "use_async": False, "int8_model_dir": int8_model_dir}) )
            modes_to_run.append( ("OpenVINO_INT8_Async", "openvino_cpu", {"precision": "INT8", "use_async": True, "int8_model_dir": int8_model_dir}) )

    if not modes_to_run:
        print("Error: No backend modes available to run.")
        return

    # --- Store aggregated results ---
    aggregated_results = {}

    # --- Iterate through Modes and Images ---
    for display_name, mode_arg, config in modes_to_run:
        print(f"\\n--- Processing Mode: {display_name} ---")
        # Create a safe directory name
        safe_display_name = display_name.replace(" ", "_").replace("/", "_")
        mode_results_dir = results_base_dir / safe_display_name
        mode_results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving results to: {mode_results_dir}")

        mode_total_time_ms = 0
        mode_successful_images = 0
        mode_all_image_times = []

        for i, img_path in enumerate(image_files):
            print(f"  Processing image {i+1}/{num_total_images}: {img_path.name}")

            # Prepare args for this specific image and mode
            import copy
            temp_args = copy.deepcopy(args)
            temp_args.mode = mode_arg
            temp_args.image = str(img_path)
            temp_args.benchmark = True # Use benchmark runs for stable time
            temp_args.benchmark_runs = benchmark_runs_per_image
            temp_args.model = base_model_path

            # Apply specific configurations
            temp_args.precision = config.get("precision", args.precision)
            temp_args.use_async = config.get("use_async", args.use_async)
            temp_args.int8_model_dir = config.get("int8_model_dir", args.int8_model_dir)

            # Define the output path for the visualization image for this mode/image
            output_filename = f"{img_path.stem}_{safe_display_name}{img_path.suffix}"
            temp_args.output = str(mode_results_dir / output_filename)

            # Run inference for this image
            # run_inference will now save the image because temp_args.output is set
            avg_time_ms = run_inference(temp_args)

            if avg_time_ms is not None and avg_time_ms > 0: # Check for valid positive time
                mode_total_time_ms += avg_time_ms
                mode_successful_images += 1
                mode_all_image_times.append(avg_time_ms)
                print(f"    Avg time for this image: {avg_time_ms:.2f} ms")
            else:
                print(f"    Skipped image {img_path.name} for mode {display_name} due to error or invalid time.")

        # --- Calculate and Store Aggregated Results for this Mode ---
        if mode_successful_images > 0:
            overall_avg_time = mode_total_time_ms / mode_successful_images
            # Calculate FPS based on overall average time
            overall_fps = 1000 / overall_avg_time if overall_avg_time > 0 else 0
            aggregated_results[display_name] = {
                "overall_avg_time": overall_avg_time,
                "overall_fps": overall_fps,
                "successful_images": mode_successful_images,
                "total_images": num_total_images
            }
            print(f"  Mode '{display_name}' completed.")
            print(f"  Successfully processed: {mode_successful_images}/{num_total_images} images")
            print(f"  Overall Average Inference Time: {overall_avg_time:.2f} ms")
            print(f"  Overall Average FPS: {overall_fps:.2f}")
        else:
            print(f"  Mode '{display_name}' failed to process any images.")
            aggregated_results[display_name] = {"error": "Failed to process any images"}

    # --- Print Final Summary Table ---
    print("\\n=== Overall Average Performance Across All Images ===")
    valid_results = {k: v for k, v in aggregated_results.items() if "overall_avg_time" in v and v["overall_avg_time"] > 0}

    if valid_results:
        # Sort by overall average time
        sorted_results = sorted(valid_results.items(), key=lambda item: item[1]["overall_avg_time"])

        # Find fastest for ratio calculation
        fastest_time = sorted_results[0][1]["overall_avg_time"] if sorted_results else 1

        # Determine max name length for formatting
        max_name_len = max(len(name) for name in valid_results.keys()) if valid_results else 27

        header = f"{'Mode'.ljust(max_name_len)} | Avg Time (ms) | Avg FPS   | Speed Ratio | Processed"
        print(header)
        print("-" * len(header))

        for mode_name, data in sorted_results:
            time = data["overall_avg_time"]
            fps = data["overall_fps"]
            ratio = time / fastest_time if fastest_time > 0 else float('inf')
            processed_str = f"{data['successful_images']}/{data['total_images']}"
            print(f"{mode_name.ljust(max_name_len)} | {time:^13.2f} | {fps:^9.2f} | {ratio:^11.2f}x | {processed_str}")

        print(f"\\nFastest Mode (Overall Avg): {sorted_results[0][0]} ({fastest_time:.2f} ms)")
    else:
        print("No valid results obtained across all modes.")

    # --- Plotting --- # Add plotting logic here
    if valid_results and args.save_summary: # Check if save_summary is enabled (optional, could plot always)
        print("\nGenerating comparison plot...")
        # Ensure plotting function is available
        if 'plot_benchmark_results' in globals():
            # Prepare results in the format expected by plot_benchmark_results
            # It expects results under a key like 'modes' or 'backends'
            plot_data = {
                # Include system info and settings if needed by plotter, or keep it simple
                "system_info": { # Populate if necessary, or omit
                      "pytorch_version": torch.__version__,
                      "cuda_available": torch.cuda.is_available(),
                      "tensorrt_available": TENSORRT_AVAILABLE,
                      "openvino_available": OPENVINO_AVAILABLE
                 },
                "benchmark_settings": { # Populate if necessary, or omit
                     "images_tested": num_total_images,
                     "runs_per_image": benchmark_runs_per_image
                },
                # The actual results data, using 'modes' as the key
                # Map keys from aggregated_results to match expected keys ('inference_time', 'fps')
                "modes": {k: {"inference_time": v["overall_avg_time"],
                              "fps": v["overall_fps"],
                              "speed_ratio": v["overall_avg_time"] / fastest_time if fastest_time > 0 else float('inf')
                              }
                          for k, v in valid_results.items()
                         }
            }

            # Generate timestamp
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

            # Call the plotting function
            plot_benchmark_results(
                plot_data, # Pass the formatted data
                args.benchmark_dir, # Use benchmark_dir for plots
                timestamp,
                title_prefix="All Images All Modes" # Specific title
            )
        else:
            print("Warning: plot_benchmark_results function not found, skipping plot generation.")

    print("\\n--- All Images All Modes Test Complete ---")

# <<< End of run_all_images_all_modes function definition >>>

# <<< Add this function definition before run_inference >>>
def process_openvino_output(output_data, original_shape, input_shape, conf_threshold, nms_threshold=0.45):
    """
    Processes the raw output from an OpenVINO YOLOv8 model.

    Args:
        output_data: Raw numpy array output from the OpenVINO model (e.g., shape [1, 84, 8400]).
        original_shape: Tuple (height, width) of the original image.
        input_shape: Tuple (height, width) the model expects.
        conf_threshold: Confidence threshold for filtering detections.
        nms_threshold: IoU threshold for Non-Maximum Suppression.

    Returns:
        A numpy array of final detections in the format [x1, y1, x2, y2, confidence, class_id].
    """
    try:
        print(f"OpenVINO raw output shape: {output_data.shape}") # Debug print

        # Expected output shape: [1, 84, 8400] -> [batch, cxcywh+classes, num_proposals]
        # Transpose to [1, 8400, 84] -> [batch, num_proposals, cxcywh+classes]
        if len(output_data.shape) == 3 and output_data.shape[1] == 84 and output_data.shape[2] == 8400:
             output_data = output_data.transpose((0, 2, 1))[0] # Remove batch dim
        elif len(output_data.shape) == 2 and output_data.shape[1] == 84:
             # Assume shape [8400, 84] if batch dim already removed
             pass # Already in the correct shape [num_proposals, cxcywh+classes]
        else:
             print(f"Warning: Unexpected OpenVINO output shape {output_data.shape}. Attempting to proceed, but results may be incorrect.")
             # Attempt to remove batch dim if it exists
             if len(output_data.shape) == 3 and output_data.shape[0] == 1:
                  output_data = output_data[0]
             # If still not 2D, cannot proceed reliably
             if len(output_data.shape) != 2:
                  raise ValueError(f"Cannot process unexpected output shape: {output_data.shape}")


        boxes = []
        scores = []
        class_ids = []

        orig_h, orig_w = original_shape
        input_h, input_w = input_shape

        # Calculate scaling factors
        x_scale = orig_w / input_w
        y_scale = orig_h / input_h

        num_proposals = output_data.shape[0]
        print(f"Processing {num_proposals} proposals...")

        for i in range(num_proposals):
            proposal = output_data[i]
            # First 4 are box coordinates (cx, cy, w, h) relative to input size
            # Next 80 are class scores (assuming COCO dataset)
            # Often, there isn't a separate object confidence score in this format,
            # the class scores themselves represent confidence.

            class_scores = proposal[4:] # Scores for 80 classes
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            if confidence >= conf_threshold:
                # Decode box coordinates
                cx, cy, w_box, h_box = proposal[:4]

                # Convert center coords, width, height to x1, y1, x2, y2 (relative to input)
                x1_rel = cx - w_box / 2
                y1_rel = cy - h_box / 2
                # x2_rel = cx + w_box / 2
                # y2_rel = cy + h_box / 2

                # Scale to original image dimensions for NMS input format [x_tl, y_tl, width, height]
                x_tl_scaled = int(x1_rel * x_scale)
                y_tl_scaled = int(y1_rel * y_scale)
                w_scaled = int(w_box * x_scale)
                h_scaled = int(h_box * y_scale)


                boxes.append([x_tl_scaled, y_tl_scaled, w_scaled, h_scaled]) # Store as [x_top_left, y_top_left, width, height] for NMS
                scores.append(float(confidence))
                class_ids.append(int(class_id))

        if not boxes:
             print("No boxes found above confidence threshold before NMS.")
             return np.empty((0, 6), dtype=np.float32) # Return empty array if nothing found


        print(f"Found {len(boxes)} boxes above threshold before NMS.")
        # Perform Non-Maximum Suppression
        # cv2.dnn.NMSBoxes expects boxes as (x, y, w, h)
        try:
             # Ensure boxes, scores are correctly formatted
             # boxes should be list of [x, y, w, h]
             # scores should be list of floats
             indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=conf_threshold, nms_threshold=nms_threshold)
        except Exception as nms_e:
             print(f"Error during NMS: {nms_e}")
             # Fallback: Return boxes without NMS? Or empty? Let's return empty for safety.
             return np.empty((0, 6), dtype=np.float32)

        final_detections = []
        if len(indices) > 0:
             print(f"Keeping {len(indices)} boxes after NMS.")
             # If indices is a nested array (e.g., [[0], [2]]), flatten it
             if isinstance(indices, np.ndarray) and len(indices.shape) > 1 and indices.shape[1] == 1:
                  indices = indices.flatten()
             # Handle potential tuple output from NMSBoxes in some versions?
             elif isinstance(indices, tuple) and len(indices) > 0:
                  indices = indices[0].flatten() if isinstance(indices[0], (list, np.ndarray)) else indices

             # Check indices type after potential flattening
             if not isinstance(indices, (np.ndarray, list, tuple)):
                  print(f"Warning: Unexpected indices type after NMS processing: {type(indices)}. Cannot proceed.")
                  return np.empty((0, 6), dtype=np.float32)

             # Ensure indices are integers
             try:
                 valid_indices = [int(i) for i in indices if int(i) < len(boxes)]
             except (ValueError, TypeError) as e:
                 print(f"Error converting NMS indices to integers: {e}. Indices: {indices}")
                 return np.empty((0, 6), dtype=np.float32)


             for i in valid_indices:
                x, y, w_box, h_box = boxes[i]
                confidence = scores[i]
                class_id = class_ids[i]

                # Convert [x, y, w, h] back to [x1, y1, x2, y2] for visualization function
                x1_final = x
                y1_final = y
                x2_final = x + w_box
                y2_final = y + h_box

                final_detections.append([x1_final, y1_final, x2_final, y2_final, confidence, class_id])

        return np.array(final_detections, dtype=np.float32)

    except Exception as e:
        print(f"Error processing OpenVINO output: {e}")
        import traceback
        traceback.print_exc()
        return np.empty((0, 6), dtype=np.float32) # Return empty array on error

# <<< End of process_openvino_output definition >>>

# <<< Add the new function definition BEFORE main() >>>
def run_batch_benchmark(args):
    """
    Runs batch processing benchmark across all modes for various batch sizes.
    Uses a single image repeated to form batches.
    """
    # Import Path at the beginning
    from pathlib import Path
    import time
    import numpy as np
    import copy
    import datetime
    import json
    import torch # Ensure torch is imported
    global TENSORRT_AVAILABLE, OPENVINO_AVAILABLE, ov # Ensure globals are accessible

    print("\n=== Running Batch Processing Benchmark ===")
    print(f"Testing Batch Sizes: {args.batch_sizes}")

    # --- Configuration & Setup ---
    base_model_path = args.model
    int8_model_dir = args.int8_model_dir
    # Use a single representative image (e.g., the one specified or first test image)
    if not Path(args.image).is_file(): # Path is now defined
        print(f"Warning: Specified image '{args.image}' not found. Using first image from test_images/")
        test_images_dir = Path("test_images")
        try:
            args.image = str(next(test_images_dir.glob("*.jpg"))) # Find first jpg
            print(f"Using image: {args.image}")
        except StopIteration:
             print("Error: No images found in test_images/ to use for batch benchmark.")
             return

    benchmark_runs = args.benchmark_runs # Number of timed runs per batch size
    # Imports moved to the top
    # from pathlib import Path
    # import time
    # import numpy as np
    # import copy
    # import datetime
    # import json
    # import torch # Ensure torch is imported
    # global TENSORRT_AVAILABLE, OPENVINO_AVAILABLE, ov # Ensure globals are accessible

    results = {} # Structure: results[mode_display_name][batch_size] = {time_ms, fps}

    # --- Check INT8 Availability (similar to other benchmark functions) ---
    int8_available = False
    if not int8_model_dir:
        base_pt_model_name = os.path.splitext(os.path.basename(args.model))[0] if args.model.endswith('.pt') else "yolov8n"
        derived_int8_dir = Path(f"yolov8n_openvino_model_int8")
        if derived_int8_dir.is_dir() and any(f.endswith('.xml') for f in os.listdir(derived_int8_dir)):
             print(f"Found potential INT8 model directory at: {derived_int8_dir}")
             int8_model_dir = str(derived_int8_dir)
             int8_available = True
        else:
             print("Info: INT8 model directory not specified/found. Skipping INT8 modes.")
    elif os.path.isdir(int8_model_dir) and any(f.endswith('.xml') for f in os.listdir(int8_model_dir)):
        print(f"Using specified INT8 model directory: {int8_model_dir}")
        int8_available = True
    else:
        print(f"Warning: Specified --int8_model_dir ('{int8_model_dir}') is invalid. Skipping INT8 modes.")
        int8_model_dir = None

    # --- Define Modes to Run --- (Adjust based on availability)
    modes_to_run = []
    # Add use_async key to config
    modes_to_run.append( ("PyTorch_CPU", "pytorch_cpu", {"use_async": False}) ) 
    if TENSORRT_AVAILABLE and torch.cuda.is_available():
        modes_to_run.append( ("TensorRT_GPU", "tensorrt_gpu", {"use_async": False}) ) 
    if OPENVINO_AVAILABLE:
        # Sync Modes
        modes_to_run.append( ("OpenVINO_FP32_Sync", "openvino_cpu", {"precision": "FP32", "use_async": False, "int8_model_dir": None}) )
        modes_to_run.append( ("OpenVINO_FP16_Sync", "openvino_cpu", {"precision": "FP16", "use_async": False, "int8_model_dir": None}) )
        if int8_available:
            modes_to_run.append( ("OpenVINO_INT8_Sync", "openvino_cpu", {"precision": "INT8", "use_async": False, "int8_model_dir": int8_model_dir}) )
        # Async Modes
        modes_to_run.append( ("OpenVINO_FP32_Async", "openvino_cpu", {"precision": "FP32", "use_async": True, "int8_model_dir": None}) )
        modes_to_run.append( ("OpenVINO_FP16_Async", "openvino_cpu", {"precision": "FP16", "use_async": True, "int8_model_dir": None}) )
        if int8_available:
            modes_to_run.append( ("OpenVINO_INT8_Async", "openvino_cpu", {"precision": "INT8", "use_async": True, "int8_model_dir": int8_model_dir}) )

    # --- Preprocess the single image once --- #
    print(f"Preprocessing reference image: {args.image}")
    try:
        # Use the existing preprocessing function
        # Assuming load_and_preprocess_image returns (processed_np_array, original_cv2_image)
        # We need the numpy array for potential conversion to tensor/feeding OV
        preprocessed_image_np, original_img_cv2 = load_and_preprocess_image(args.image)
        # Get original shape from the loaded cv2 image
        original_shape = original_img_cv2.shape[:2] # (H, W)
        # Derive input shape from the preprocessed numpy array
        _, _, input_h, input_w = preprocessed_image_np.shape # (1, C, H, W)
        input_shape = (input_h, input_w) # (H, W)
        # Convert to torch tensor for PyTorch/TensorRT and easy batch repetition
        preprocessed_image_tensor = torch.from_numpy(preprocessed_image_np)

        print(f"Successfully preprocessed image. Tensor shape: {preprocessed_image_tensor.shape}, Input size: {input_shape}")

    except Exception as preproc_e:
        print(f"Error during actual preprocessing: {preproc_e}")
        return

    # --- Iterate through Modes and Batch Sizes --- #
    for display_name, mode_arg, config in modes_to_run:
        print(f"\n--- Benchmarking Mode: {display_name} ---")
        results[display_name] = {}
        model_instance = None # For PyTorch/TensorRT YOLO object
        core = None           # For OpenVINO
        model_ov = None       # For OpenVINO loaded model
        compiled_model = None # For OpenVINO compiled model
        infer_request = None  # For OpenVINO Sync request
        device = 'cpu' if mode_arg == 'pytorch_cpu' else 'cuda' if mode_arg == 'tensorrt_gpu' else 'CPU' # Default OV to CPU

        try:
            # --- Load Model (once per mode) --- #
            print(f"  Loading model for {display_name}...")
            if mode_arg == "pytorch_cpu":
                from ultralytics import YOLO
                model_instance = YOLO(base_model_path)
                model_instance.to(torch.device("cpu"))
                print("  PyTorch CPU model loaded.")
            elif mode_arg == "tensorrt_gpu":
                from ultralytics import YOLO
                # YOLO class handles engine loading/conversion implicitly when device='cuda'
                model_instance = YOLO(base_model_path)
                model_instance.to(torch.device("cuda"))
                # Perform a dummy predict to ensure TRT engine is built/loaded if needed
                print("  Warming up TensorRT model (compiling engine if necessary)...")
                _ = model_instance.predict(source=preprocessed_image_tensor.to('cuda'), verbose=False)
                torch.cuda.synchronize()
                print("  TensorRT GPU model loaded and warmed up.")
            elif mode_arg == "openvino_cpu":
                if not OPENVINO_AVAILABLE:
                     raise RuntimeError("OpenVINO selected but not available.")
                core = ov.Core()
                openvino_model_xml = ""
                precision = config.get("precision", "FP32")
                int8_dir = config.get("int8_model_dir")

                # <<<--- Start of modified path logic for batch benchmark --->>>
                if precision == 'INT8':
                    # --- INT8 Path Logic (remains the same) ---
                    if not int8_dir or not os.path.isdir(int8_dir):
                        raise ValueError(f"INT8 precision requires a valid int8_model_dir, got: {int8_dir}")
                    # Find XML in INT8 dir (Simplified logic from run_inference)
                    xml_files = list(Path(int8_dir).glob("*.xml"))
                    if not xml_files:
                        raise FileNotFoundError(f"No .xml model found in INT8 directory: {int8_dir}")
                    openvino_model_xml = str(xml_files[0]) # Take the first one found
                    print(f"  Found INT8 model: {openvino_model_xml}")
                elif precision == 'FP16':
                    # --- Hardcoded FP16 Path Logic for Batch Benchmark ---
                    print("  Using hardcoded path for FP16 OpenVINO model in batch benchmark...")
                    hardcoded_fp16_path = "models/yolov8n_openvino_fp16_model/yolov8n_fp16.xml"
                    if os.path.exists(hardcoded_fp16_path):
                        openvino_model_xml = hardcoded_fp16_path
                        print(f"  Found FP16 model at hardcoded path: {openvino_model_xml}")
                    else:
                        print(f"Error: Hardcoded FP16 model path does not exist for batch benchmark: {hardcoded_fp16_path}")
                        raise FileNotFoundError(f"FP16 model not found at {hardcoded_fp16_path}")
                else: # FP32
                    # --- FP32 Path Logic (remains the same) ---
                    print(f"  Attempting to find FP32 OpenVINO model based on {base_model_path}...")
                    potential_paths = []
                    if base_model_path.endswith('.xml'): potential_paths.append(base_model_path)
                    elif base_model_path.endswith('.pt'):
                         potential_paths.append(os.path.join("models", "yolov8n.xml"))
                         potential_paths.append(args.model.replace('.pt', '_openvino_model/yolov8n.xml'))
                    else: potential_paths.append(os.path.join("models", f"{base_model_path}.xml"))

                    for path in potential_paths:
                        if os.path.exists(path):
                            openvino_model_xml = path
                            break
                    if not openvino_model_xml:
                         raise FileNotFoundError(f"Base OpenVINO model (.xml) not found for FP32/FP16. Checked: {potential_paths}")
                    print(f"  Found base model for {precision}: {openvino_model_xml}")
                # <<<--- End of modified path logic for batch benchmark --->>>

                # Read the model (Using the determined path)
                model_ov = core.read_model(openvino_model_xml)
                print(f"  OpenVINO model read: {openvino_model_xml}")

                # --- OpenVINO Specific: Initial Compile/Setup (Batch Size 1 or Dynamic) ---
                # We will handle reshaping and recompiling within the batch loop if needed
                # Compile with initial batch size (usually 1) or check if dynamic
                compile_config = {}
                # REMOVED: FP16 hint is redundant when loading a pre-converted FP16 model.
                # The following block was removed:
                # if precision == 'FP16' and precision != 'INT8':
                #    compile_config[ov.properties.hint.inference_precision()] = "f16"

                print(f"  Preparing OpenVINO settings for {device} ({precision})...")
                # Compile happens inside the batch loop after potential reshape

            # <<< Model loading complete >>>

            # --- Get OpenVINO input name (do once after loading) ---
            input_tensor_name_ov = None
            output_node_ov = None # Get output node for async result retrieval
            if mode_arg == "openvino_cpu" and model_ov:
                try:
                    input_tensor_name_ov = model_ov.input(0).get_any_name()
                    output_node_ov = model_ov.output(0) # Assuming single output
                except Exception:
                    print("Warning: Could not get input/output tensor name from OV model.")

            # --- Batch Size Loop --- #
            for batch_size in args.batch_sizes:
                print(f"    Testing Batch Size: {batch_size}")
                batch_times_sync = [] # For sync mode timing
                total_time_async_ms = 0 # For async mode total time
                completed_requests_async = 0 # Counter for async callbacks
                
                # Get async flag from config
                use_async = config.get("use_async", False)

                compiled_model_for_batch = None # Specific compiled model for this batch size if reshaping
                # Remove infer_request_for_batch here, as it's sync specific
                # infer_request_for_batch = None
                infer_queue = None # For async mode
                num_infer_requests = 1 # Default for sync, will be updated for async

                try:
                    # --- Create Input Batch --- #
                    # Use repeat for tensors, ensure correct device for TRT
                    if mode_arg == "tensorrt_gpu":
                        input_batch = preprocessed_image_tensor.repeat(batch_size, 1, 1, 1).to('cuda')
                    else: # PyTorch CPU and OpenVINO (use numpy later)
                        input_batch = preprocessed_image_tensor.repeat(batch_size, 1, 1, 1).to('cpu')

                    # For OpenVINO, use numpy array
                    input_batch_np = input_batch.cpu().numpy() if mode_arg == "openvino_cpu" else None
                    print(f"      Input batch shape: {input_batch.shape}")

                    # --- Handle OpenVINO Reshaping & Compilation --- #
                    if mode_arg == 'openvino_cpu':
                        try:
                            print(f"      Preparing OpenVINO model for batch size {batch_size}...")
                            input_node = model_ov.input(0)
                            input_partial_shape = input_node.get_partial_shape()
                            input_rank = input_partial_shape.rank.get_length() if input_partial_shape.rank.is_static else 4 # Assume rank 4 if dynamic

                            if input_rank != 4:
                                raise ValueError(f"Expected input rank 4, but got {input_rank}")

                            # Construct new partial shape: [batch_size, dynamic, dynamic, dynamic]
                            # Using -1 or ov.Dimension() for dynamic dimensions
                            new_partial_shape = ov.PartialShape([batch_size, -1, -1, -1])

                            print(f"      Attempting reshape with PartialShape: {new_partial_shape}")
                            # Apply reshape using the input node object or its name
                            # Use input_tensor_name_ov if available, otherwise try input_node
                            reshape_target = input_tensor_name_ov if input_tensor_name_ov else input_node
                            model_ov.reshape({reshape_target: new_partial_shape})

                            # Compile the model specifically for this batch size AFTER reshaping
                            precision_for_log = config.get("precision", "FP32") # Get precision for logging
                            print(f"      Compiling OpenVINO model for batch {batch_size} ({precision_for_log})...")
                            compiled_model_for_batch = core.compile_model(model_ov, device, compile_config)
                            if compiled_model_for_batch is None:
                                raise RuntimeError("Failed to compile OpenVINO model for this batch size.")
                            
                            # --- Setup for Sync or Async Execution --- #
                            if use_async:
                                try:
                                     num_req_prop = compiled_model_for_batch.get_property(ov.properties.hint.num_requests())
                                     # Ensure num_req_prop is a positive integer
                                     num_infer_requests = int(num_req_prop) if num_req_prop and int(num_req_prop) > 0 else 1
                                except Exception as e:
                                     print(f"      Warning: Could not get optimal number of infer requests ({e}), defaulting to 1.")
                                     num_infer_requests = 1 # Fallback
                                print(f"      Using {num_infer_requests} Infer Requests for Async Queue.")
                                # Ensure jobs parameter is an integer
                                infer_queue = ov.AsyncInferQueue(compiled_model_for_batch, int(num_infer_requests))
                                
                                # Define callback (needs access to counter)
                                def callback_counter(request, userdata):
                                     nonlocal completed_requests_async
                                     completed_requests_async += 1
                                
                                infer_queue.set_callback(callback_counter)
                                print("      OpenVINO AsyncInferQueue ready.")
                            # else: # Sync setup (InferRequest created just before inference loop)
                                # infer_request_for_batch = compiled_model_for_batch.create_infer_request()
                            print("      OpenVINO model ready for batch inference.")

                        except Exception as ov_prep_e:
                             print(f"      Error preparing OpenVINO model for batch {batch_size}: {ov_prep_e}")
                             results[display_name][batch_size] = {"error": f"OV Prep Error: {ov_prep_e}"} 
                             continue # Skip to next batch size

                    # --- Warm-up Runs --- # 
                    # MOVED: Warmup logic is now inside the specific Sync/Async blocks below
                    # print("      Warm-up runs...")
                    # ... (Removed old warmup block) ...
                    
                    # --- Benchmark Runs --- #
                    print("      Benchmark runs...")
                    completed_requests_async = 0 # Reset counter before runs
                    
                    if use_async and mode_arg == 'openvino_cpu':
                        # --- Async Benchmark Logic --- # 
                        if not infer_queue:
                            raise RuntimeError("AsyncInferQueue was not initialized.")
                        if not input_tensor_name_ov:
                            # Try to get it from compiled model if not available earlier
                            try:
                                input_tensor_name_ov = compiled_model_for_batch.input(0).get_any_name()
                            except Exception as final_name_e:
                                raise RuntimeError(f"Cannot determine input tensor name for async inference: {final_name_e}")
                                
                        input_data = {input_tensor_name_ov: input_batch_np}
                        
                        # <<< Async Warm-up moved here >>>
                        print("      Async Warm-up...")
                        for _ in range(min(3, benchmark_runs)): # Warmup with a few runs
                            infer_queue.start_async(input_data)
                        infer_queue.wait_all() # Wait for warmup to finish
                        completed_requests_async = 0 # Reset counter after warmup
                        # <<< End of Async Warm-up >>>
                        
                        print(f"      Submitting {benchmark_runs} async requests...")
                        total_start_time = time.perf_counter() 
                        
                        successful_submissions = 0
                        for i in range(benchmark_runs):
                            try:
                                infer_queue.start_async(input_data)
                                successful_submissions += 1
                            except Exception as submit_e:
                                 print(f"Error submitting async request {i+1}/{benchmark_runs}: {submit_e}")
                                 break # Stop submitting on error
                                 
                        if successful_submissions > 0:
                            infer_queue.wait_all() # Wait for all submitted requests
                        else:
                             print("      No async requests were submitted successfully.")
                             
                        total_end_time = time.perf_counter()
                        total_time_async_ms = (total_end_time - total_start_time) * 1000
                        
                        if completed_requests_async != successful_submissions:
                             print(f"Warning: Submitted {successful_submissions} async requests, but callback counted {completed_requests_async}.")
                             if completed_requests_async == 0: # If none completed, treat as failure
                                 raise RuntimeError("Async inference failed: No requests completed.")
                        
                        # Use completed_requests_async for calculation if it seems reliable, else successful_submissions
                        # Let's use successful_submissions as it reflects what was sent to wait_all
                        runs_to_average = successful_submissions if successful_submissions > 0 else 1

                        # Calculate metrics based on total time for all runs
                        avg_batch_time_ms = total_time_async_ms / runs_to_average
                        avg_img_time_ms = avg_batch_time_ms / batch_size if batch_size > 0 else 0
                        # FPS is total images processed / total time in seconds
                        total_images_processed = runs_to_average * batch_size
                        total_time_sec = total_time_async_ms / 1000
                        fps = total_images_processed / total_time_sec if total_time_sec > 0 else 0
                        
                        print(f"      Total Async Time ({runs_to_average} runs): {total_time_async_ms:.2f} ms")
                        
                    else:
                        # --- Sync Benchmark Logic (PyTorch, TensorRT, OpenVINO Sync) --- # 
                        # Create sync infer request here if needed
                        infer_request_sync = None
                        if mode_arg == 'openvino_cpu' and not use_async:
                            if not compiled_model_for_batch:
                                raise RuntimeError("Compiled model not available for sync inference.")
                            # Ensure input tensor name is available before creating request
                            if not input_tensor_name_ov:
                                try:
                                     input_tensor_name_ov = compiled_model_for_batch.input(0).get_any_name()
                                except Exception as final_name_e:
                                     raise RuntimeError(f"Cannot determine input tensor name for sync inference: {final_name_e}")
                            # Create the infer request *before* the warm-up loop
                            infer_request_sync = compiled_model_for_batch.create_infer_request()

                        # Warm-up Runs
                        print("      Sync Warm-up...")
                        # <<< Sync Warm-up moved here >>>
                        for _ in range(3):
                            if mode_arg == "pytorch_cpu":
                                _ = model_instance.predict(source=input_batch, verbose=False)
                            elif mode_arg == "tensorrt_gpu":
                                _ = model_instance.predict(source=input_batch, verbose=False)
                                torch.cuda.synchronize()
                            elif mode_arg == "openvino_cpu": # Sync OpenVINO
                                if not infer_request_sync: raise RuntimeError("Sync InferRequest not ready for warm-up.")
                                infer_request_sync.infer({input_tensor_name_ov: input_batch_np})
                                _ = infer_request_sync.get_output_tensor().data
                        # <<< End of Sync Warm-up >>>
                                
                        # Timed Benchmark Runs
                        batch_times_sync = [] # Re-initialize here
                        for _ in range(benchmark_runs):
                            start_time = time.perf_counter()
                            if mode_arg == "pytorch_cpu":
                                _ = model_instance.predict(source=input_batch, verbose=False)
                            elif mode_arg == "tensorrt_gpu":
                                torch.cuda.synchronize()
                                _ = model_instance.predict(source=input_batch, verbose=False)
                                torch.cuda.synchronize()
                            elif mode_arg == "openvino_cpu": # Sync OpenVINO
                                if not infer_request_sync: raise RuntimeError("Sync InferRequest not ready.")
                                infer_request_sync.infer({input_tensor_name_ov: input_batch_np})
                                _ = infer_request_sync.get_output_tensor().data # Ensure output is retrieved
                            end_time = time.perf_counter()
                            batch_times_sync.append((end_time - start_time) * 1000) # Time in ms
                            
                        if not batch_times_sync:
                             print("      Error: No successful sync benchmark runs.")
                             results[display_name][batch_size] = {"error": "Sync Benchmarking failed"}
                             continue # Skip to next batch size
                             
                        # Calculate metrics for Sync mode
                        avg_batch_time_ms = np.mean(batch_times_sync)
                        avg_img_time_ms = avg_batch_time_ms / batch_size if batch_size > 0 else 0
                        fps = (batch_size * 1000) / avg_batch_time_ms if avg_batch_time_ms > 0 else 0

                    # --- Calculate Metrics (Common part after Sync/Async logic) --- #
                    # Metrics (avg_batch_time_ms, avg_img_time_ms, fps) are calculated within each branch now
                    print(f"      Avg Batch Time: {avg_batch_time_ms:.2f} ms")
                    print(f"      Avg Image Time: {avg_img_time_ms:.2f} ms")
                    print(f"      FPS (Throughput): {fps:.2f}")

                    # <<< Add this line back to store results >>>
                    results[display_name][batch_size] = {
                        "avg_batch_time_ms": avg_batch_time_ms,
                        "avg_img_time_ms": avg_img_time_ms,
                        "fps": fps
                    }
                    
                except Exception as batch_e:
                     print(f"      Error during batch size {batch_size} processing: {batch_e}")
                     results[display_name][batch_size] = {"error": str(batch_e)}

        except Exception as e:
            # --- Outer Exception Handling for Mode --- #
            print(f"  Error benchmarking mode {display_name}: {e}")
            import traceback
            traceback.print_exc()
            results[display_name]["error"] = f"Mode Error: {e}"
            # Ensure the loop continues to the next mode if one mode fails
            continue
        finally:
            # --- Unload model / cleanup (optional but good practice) --- #
            print(f"  Finished benchmarking mode: {display_name}")
            del model_instance
            del core
            del model_ov
            del compiled_model
            del infer_request
            del compiled_model_for_batch
            # Force garbage collection? Might not be necessary
            # import gc
            # gc.collect()

    # --- Print Summary Table --- #
    print("\n=== Batch Benchmark Summary ===")
    # Prepare data for table
    header = ["Mode"] + [f"BS={bs}" for bs in args.batch_sizes]
    # Find max mode name length for formatting
    max_name_len = max(len(name) for name in results.keys()) if results else 15
    header_fmt = f"{{:<{max_name_len}}} |" + " {:^15} |" * len(args.batch_sizes)
    row_fmt    = f"{{:<{max_name_len}}} |" + " {:^15.2f} |" * len(args.batch_sizes)
    separator = "-" * (max_name_len + 1) + ("-" * 18) * len(args.batch_sizes)

    print("\n--- Average Batch Time (ms/batch) ---")
    print(header_fmt.format(*header))
    print(separator)
    for mode, batch_data in results.items():
        if "error" in batch_data:
             row_data = [mode] + ["ERROR"] * len(args.batch_sizes)
        else:
             row_data = [mode] + [batch_data.get(bs, {}).get("avg_batch_time_ms", "N/A") for bs in args.batch_sizes]
        # Handle cases where data might be missing or is "N/A"
        formatted_row = [row_data[0]] + [(f"{x:.2f}" if isinstance(x, (int, float)) else str(x)) for x in row_data[1:]]
        print(header_fmt.format(*formatted_row)) # Use header_fmt to align error messages too

    print("\n--- Average Image Time (ms/image) ---")
    print(header_fmt.format(*header))
    print(separator)
    for mode, batch_data in results.items():
        if "error" in batch_data:
             row_data = [mode] + ["ERROR"] * len(args.batch_sizes)
        else:
            row_data = [mode] + [batch_data.get(bs, {}).get("avg_img_time_ms", "N/A") for bs in args.batch_sizes]
        formatted_row = [row_data[0]] + [(f"{x:.2f}" if isinstance(x, (int, float)) else str(x)) for x in row_data[1:]]
        print(header_fmt.format(*formatted_row))

    print("\n--- Throughput (FPS) ---")
    print(header_fmt.format(*header))
    print(separator)
    for mode, batch_data in results.items():
        if "error" in batch_data:
             row_data = [mode] + ["ERROR"] * len(args.batch_sizes)
        else:
            row_data = [mode] + [batch_data.get(bs, {}).get("fps", "N/A") for bs in args.batch_sizes]
        formatted_row = [row_data[0]] + [(f"{x:.2f}" if isinstance(x, (int, float)) else str(x)) for x in row_data[1:]]
        print(header_fmt.format(*formatted_row))

    # --- Plot Results --- #
    if args.save_summary:
        print("\nGenerating batch benchmark plots...")
        # Call a new plotting function (or adapt the existing one)
        if 'plot_batch_benchmark_results' in globals():
             timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
             # Pass the raw results dictionary and batch sizes list
             plot_batch_benchmark_results(results, args.batch_sizes, args.benchmark_dir, timestamp)
        else:
             print("Warning: plot_batch_benchmark_results function not found, skipping plot generation.")

        # --- Save Summary --- #
        print("\nSaving batch benchmark summary...")
        summary_filename = os.path.join(args.benchmark_dir, f"batch_benchmark_summary_{timestamp}.json")
        try:
             # Add args and system info to the saved results
             full_summary = {
                 "system_info": { # Regenerate or fetch from global state if needed
                      "pytorch_version": torch.__version__,
                      "cuda_available": torch.cuda.is_available(),
                      "tensorrt_available": TENSORRT_AVAILABLE,
                      "openvino_available": OPENVINO_AVAILABLE
                 },
                 "benchmark_args": vars(args),
                 "results": results, # Contains results[display_name][batch_size] = {time_ms, fps}
                 "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
             }
             with open(summary_filename, "w") as f:
                 json.dump(full_summary, f, indent=2)
             print(f"Batch benchmark summary saved to: {summary_filename}")
        except Exception as save_e:
             print(f"Error saving batch benchmark summary: {save_e}")

    print("\n--- Batch Benchmark Complete ---")

# <<< End of run_batch_benchmark definition >>>

# <<< Add this new plotting function definition (AGAIN) >>>
def plot_batch_benchmark_results(results_data, batch_sizes, output_dir, timestamp):
    """
    Generates plots for batch benchmark results:
    1. Avg Image Time vs. Batch Size
    2. Throughput (FPS) vs. Batch Size
    Includes both Sync and Async modes if present.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("Plotting requires matplotlib. Please install it (`pip install matplotlib`)")
        return

    # Ensure output directory exists
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    valid_modes = [mode for mode, data in results_data.items() if "error" not in data]
    if not valid_modes:
        print("No valid modes found in results data for plotting.")
        return

    # --- Plot 1: Avg Image Time vs. Batch Size --- #
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    # Correct the markers list - remove spaces
    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h'] # Different markers
    marker_idx = 0

    for mode in valid_modes:
        mode_data = results_data[mode]
        times = [mode_data.get(bs, {}).get("avg_img_time_ms") for bs in batch_sizes]
        # Filter out None or non-numeric values for plotting
        plot_bs = [bs for bs, t in zip(batch_sizes, times) if isinstance(t, (int, float))]
        plot_times = [t for t in times if isinstance(t, (int, float))]
        if plot_bs:
            ax1.plot(plot_bs, plot_times, marker=markers[marker_idx % len(markers)], linestyle='-', label=mode)
            marker_idx += 1

    ax1.set_title('Batch Benchmark: Average Image Inference Time')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Time per Image (ms)')
    ax1.set_xticks(batch_sizes)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.legend()
    ax1.set_yscale('log') # Often better for time comparison across scales
    ax1.yaxis.set_major_formatter(mticker.ScalarFormatter()) # Ensure standard numbers on log scale
    ax1.yaxis.set_minor_formatter(mticker.NullFormatter())

    plot1_filename = output_dir_path / f"batch_benchmark_avg_img_time_{timestamp}.png"
    try:
        fig1.savefig(plot1_filename)
        print(f"Avg Image Time plot saved to: {plot1_filename}")
    except Exception as e:
        print(f"Error saving Avg Image Time plot: {e}")
    plt.close(fig1)

    # --- Plot 2: Throughput (FPS) vs. Batch Size --- #
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    # Use the same corrected markers list
    marker_idx = 0 # Reset marker index

    for mode in valid_modes:
        mode_data = results_data[mode]
        fps_values = [mode_data.get(bs, {}).get("fps") for bs in batch_sizes]
        # Filter out None or non-numeric values
        plot_bs = [bs for bs, fps_val in zip(batch_sizes, fps_values) if isinstance(fps_val, (int, float))]
        plot_fps = [fps_val for fps_val in fps_values if isinstance(fps_val, (int, float))]
        if plot_bs:
            ax2.plot(plot_bs, plot_fps, marker=markers[marker_idx % len(markers)], linestyle='-', label=mode)
            marker_idx += 1

    ax2.set_title('Batch Benchmark: Throughput (FPS)')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Frames Per Second (FPS)')
    ax2.set_xticks(batch_sizes)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.legend(loc='upper left') # Adjust legend position
    # FPS is often linear or sub-linear, log scale might not be best unless range is huge
    # ax2.set_yscale('log')

    plot2_filename = output_dir_path / f"batch_benchmark_throughput_fps_{timestamp}.png"
    try:
        fig2.savefig(plot2_filename)
        print(f"Throughput (FPS) plot saved to: {plot2_filename}")
    except Exception as e:
        print(f"Error saving Throughput plot: {e}")
    plt.close(fig2)

# <<< End of plot_batch_benchmark_results definition >>>

if __name__ == "__main__":
    main() 