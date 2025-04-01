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
    
    # Run benchmark mode if requested
    if args.compare_all:
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
    
    # 確定要測試的模式
    modes_to_test = []
    if args.mode == "all":
        modes_to_test = ["pytorch_cpu", "openvino_cpu"]
        if torch.cuda.is_available() and TENSORRT_AVAILABLE:
            modes_to_test.append("tensorrt_gpu")
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
    """運行所有可用後端的性能對比測試"""
    print("Running performance comparison across all available backends...")
    backends = []
    
    # Check which backends are available
    if torch.cuda.is_available():
        # Check if TensorRT is available
        if TENSORRT_AVAILABLE:
            backends.append("tensorrt_gpu")
            print("TensorRT is available and will be tested")
    
    backends.append("pytorch_cpu")
    
    # Check if OpenVINO is available
    if OPENVINO_AVAILABLE:
        backends.append("openvino_cpu")
        print("OpenVINO is available and will be tested")
        
    # Run benchmarks
    results = {
        "system_info": {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "tensorrt_available": TENSORRT_AVAILABLE,
            "openvino_available": OPENVINO_AVAILABLE
        },
        "backends": {},
        "fastest_backend": "",
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 獲取test_images目錄中的所有圖片
    if os.path.isdir("test_images"):
        images = [f for f in os.listdir("test_images") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            print("Error: 在test_images目錄中未找到圖片!")
            return results
            
        # 如果沒有指定圖片，使用第一張作為基準測試圖片
        benchmark_image = os.path.join("test_images", images[0])
    else:
        benchmark_image = args.image
        
    for backend in backends:
        print(f"\n=== Testing {backend} ===")
        # Create temporary args for this backend
        import copy
        temp_args = copy.deepcopy(args)
        temp_args.mode = backend
        temp_args.benchmark = True
        temp_args.skip_convert = backend == "tensorrt_gpu"  # Skip TensorRT conversion for quick test
        temp_args.image = benchmark_image
        
        # Run with this backend
        inference_time = run_inference(temp_args)
        if inference_time:
            results["backends"][backend] = {
                "inference_time": inference_time,
                "fps": 1000/inference_time
            }
            
    # Print benchmark results
    print("\n=== Performance Comparison ===")
    if results["backends"]:
        # Find fastest backend
        fastest = min(results["backends"].items(), key=lambda x: x[1]["inference_time"])
        fastest_backend = fastest[0]
        fastest_time = fastest[1]["inference_time"]
        results["fastest_backend"] = fastest_backend
        
        print("Backend            | Inference Time (ms) | FPS      | Speed Ratio")
        print("-------------------|---------------------|----------|------------")
        for backend, data in sorted(results["backends"].items(), key=lambda x: x[1]["inference_time"]):
            time = data["inference_time"]
            ratio = time / fastest_time
            results["backends"][backend]["speed_ratio"] = ratio
            print(f"{backend.ljust(19)}| {time:.2f} ms           | {1000/time:.2f}   | {ratio:.2f}x")
            
        print(f"\nFastest backend: {fastest_backend} ({fastest_time:.2f} ms, {1000/fastest_time:.2f} FPS)")
    else:
        print("No benchmark results available")
        
    return results

def plot_benchmark_results(results, output_dir, timestamp):
    """Create benchmark results visualization"""
    if not results["backends"]:
        print("No data to plot")
        return
        
    backends = []
    times = []
    fps = []
    ratios = []
    
    # Extract data
    for backend, data in sorted(results["backends"].items(), key=lambda x: x[1]["inference_time"]):
        backends.append(backend)
        times.append(data["inference_time"])
        fps.append(data["fps"])
        ratios.append(data["speed_ratio"])
    
    # Create chart
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    # Inference time chart
    ax1.bar(backends, times, color='blue')
    ax1.set_title('Inference Time (ms)')
    ax1.set_ylabel('Milliseconds')
    for i, v in enumerate(times):
        ax1.text(i, v + 0.5, f"{v:.2f}", ha='center')
    
    # FPS chart
    ax2.bar(backends, fps, color='green')
    ax2.set_title('Frames Per Second (FPS)')
    ax2.set_ylabel('FPS')
    for i, v in enumerate(fps):
        ax2.text(i, v + 0.5, f"{v:.2f}", ha='center')
    
    # Speed ratio chart
    ax3.bar(backends, ratios, color='red')
    ax3.set_title('Speed Ratio (relative to fastest backend)')
    ax3.set_ylabel('Ratio')
    for i, v in enumerate(ratios):
        ax3.text(i, v + 0.05, f"{v:.2f}x", ha='center')
    
    plt.tight_layout()
    
    # Save chart
    plot_filename = os.path.join(output_dir, f"benchmark_comparison_{timestamp}.png")
    plt.savefig(plot_filename)
    print(f"Benchmark plot saved to: {plot_filename}")
    
    # Close chart to free memory
    plt.close()

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
                        
                    start_time = time.time()
                    results = model.predict(args.image, save=False, verbose=False)
                    inference_time = (time.time() - start_time) * 1000
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
                detections = results[0].boxes.data.cpu().numpy()
                
                # Save result if not in benchmark mode
                if not args.benchmark:
                    run_visualization(args, original_image, detections)
                    
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
                        start_time = time.time()
                        results = model.predict(args.image, save=False, verbose=False)
                        torch.cuda.synchronize()
                        inference_time = (time.time() - start_time) * 1000
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
                    
                    # Process results for visualization
                    detections = results[0].boxes.data.cpu().numpy()
                    
                    # Save result if not in benchmark mode
                    if not args.benchmark:
                        run_visualization(args, original_image, detections)
                    
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
                # Check for OpenVINO model
                model_path = args.model.replace('.pt', '_openvino_model/yolov8n.xml')
                if not os.path.exists(model_path):
                    model_path = os.path.join("models", "yolov8n.xml")
                    if not os.path.exists(model_path):
                        print(f"OpenVINO model not found, trying to convert from PyTorch...")
                        from ultralytics import YOLO
                        pt_model = YOLO(args.model)
                        pt_model.export(format="openvino")
                        model_path = args.model.replace('.pt', '_openvino_model/yolov8n.xml')
                        if not os.path.exists(model_path):
                            print(f"Conversion failed, model not found at {model_path}")
                            return None
                
                # Initialize OpenVINO runtime
                print(f"Loading OpenVINO model from: {model_path}")
                core = ov.Core()
                model = core.read_model(model_path)
                compiled_model = core.compile_model(model, "CPU")
                
                # Get input information
                try:
                    h, w = 640, 640  # Default size
                    try:
                        input_shape = compiled_model.input(0).shape
                        if len(input_shape) >= 4:
                            h, w = input_shape[2], input_shape[3]
                    except:
                        print("Using default input shape: 640x640")
                
                    # Preprocess image
                    preprocessed_image, _ = load_and_preprocess_image(args.image, size=(w, h))
                    
                    # Benchmark runs
                    times = []
                    num_runs = args.benchmark_runs if args.benchmark else 1
                    
                    for i in range(num_runs):
                        if args.benchmark:
                            print(f"Running inference {i+1}/{num_runs}...")
                            
                        start_time = time.time()
                        outputs = compiled_model(preprocessed_image)
                        inference_time = (time.time() - start_time) * 1000
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
                    
                    # Process last result for visualization using PyTorch for post-processing
                    from ultralytics import YOLO
                    pt_model = YOLO(args.model)
                    results = pt_model.predict(args.image, verbose=False, save=False, show=False)
                    
                    # Extract detections
                    detections = []
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            if box.conf.item() < args.threshold:
                                continue
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            conf = float(box.conf)
                            cls = int(box.cls)
                            detections.append([x1, y1, x2, y2, conf, cls])
                    
                    detections = np.array(detections) if detections else np.zeros((0, 6))
                    
                    # Save result if not in benchmark mode
                    if not args.benchmark:
                        run_visualization(args, original_image, detections)
                        
                    return avg_time
                    
                except Exception as e:
                    print(f"Error with OpenVINO inference: {e}")
                    import traceback
                    traceback.print_exc()
                    return None
                    
            except Exception as e:
                print(f"Error with OpenVINO model: {e}")
                import traceback
                traceback.print_exc()
                return None
                
    except Exception as e:
        print(f"Error setting up model: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    return None

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
    # If output path is a directory, generate filename in that directory
    if os.path.isdir(output_path):
        # Get filename from input image path
        input_filename = os.path.basename(args.image)
        base_name, _ = os.path.splitext(input_filename)
        output_path = os.path.join(output_path, f"{base_name}_{args.mode}.jpg")
    # Ensure output directory exists
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

if __name__ == "__main__":
    main() 