import os
import time
import json
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path

class TensorRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # 取得輸入輸出綁定索引和形狀資訊
        self.input_idx = 0
        self.output_idx = 1
        self.batch_size = 1
        
    def load_engine(self, engine_path):
        print(f"正在載入TensorRT引擎: {engine_path}")
        with open(engine_path, 'rb') as f:
            return self.runtime.deserialize_cuda_engine(f.read())
            
    def allocate_buffers(self, batch_size):
        """分配CUDA記憶體"""
        self.batch_size = batch_size
        
        # 取得輸入形狀
        input_shape = (batch_size, 3, 640, 640)
        input_size = np.prod(input_shape) * np.dtype(np.float32).itemsize
        
        # 分配輸入和輸出記憶體
        self.d_input = cuda.mem_alloc(input_size)
        self.d_output = cuda.mem_alloc(10 * 1024 * 1024)  # 分配足夠大的輸出緩衝區
        
        # 綁定緩衝區
        self.bindings = [int(self.d_input), int(self.d_output)]
        
        # 設置輸入形狀
        if not self.engine.has_implicit_batch_dimension:
            self.context.set_binding_shape(self.input_idx, input_shape)
            
        return input_shape
    
    def infer(self, batch_data):
        """執行推理"""
        # 分配緩衝區
        input_shape = self.allocate_buffers(len(batch_data))
        
        # 預處理圖像
        input_data = np.zeros(input_shape, dtype=np.float32)
        for i, img in enumerate(batch_data):
            # 調整大小並轉換
            img = cv2.resize(img, (640, 640))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            input_data[i] = img
            
        # 複製輸入數據到GPU
        cuda.memcpy_htod_async(self.d_input, input_data, self.stream)
        
        # 運行推理
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 同步
        self.stream.synchronize()
        
        return True

def load_images(image_dir, max_images=None):
    """載入圖像"""
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    if max_images:
        image_files = image_files[:max_images]
    
    images = []
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    
    return images

def run_batch_tests(batch_sizes=[1, 2, 4, 8]):
    """運行不同批處理大小的測試"""
    engine_path = "yolov8n.engine"
    image_dir = "test_images"
    output_file = "benchmark_results/tensorrt_batch_results.json"
    warmup_runs = 3
    test_runs = 5
    
    # 載入圖像
    images = load_images(image_dir)
    print(f"已載入 {len(images)} 張圖像")
    
    # 初始化TensorRT
    try:
        trt_inference = TensorRTInference(engine_path)
    except Exception as e:
        print(f"初始化TensorRT失敗: {e}")
        return None
    
    # 測試結果
    results = {}
    
    # 為每個批處理大小運行測試
    for batch_size in batch_sizes:
        print(f"\n測試批處理大小: {batch_size}")
        
        # 創建批次
        batches = []
        for i in range(0, len(images), batch_size):
            batch = images[i:min(i+batch_size, len(images))]
            if len(batch) < batch_size:
                # 填充最後一批
                batch = batch + [images[0]] * (batch_size - len(batch))
            batches.append(batch)
        
        print(f"  創建了 {len(batches)} 個批次")
        
        # 熱身
        print("  熱身中...")
        for _ in range(warmup_runs):
            trt_inference.infer(batches[0])
        
        # 基準測試
        print("  運行基準測試...")
        batch_times = []
        for _ in range(test_runs):
            for batch in batches:
                start_time = time.time()
                trt_inference.infer(batch)
                end_time = time.time()
                batch_times.append((end_time - start_time) * 1000)  # 轉換為毫秒
        
        # 計算統計信息
        avg_time = np.mean(batch_times)
        fps = (1000 / avg_time) * batch_size
        
        results[batch_size] = {
            "avg_time_ms": float(avg_time),
            "fps": float(fps),
            "min_ms": float(np.min(batch_times)),
            "max_ms": float(np.max(batch_times)),
            "std_ms": float(np.std(batch_times))
        }
        
        print(f"  平均批處理時間: {avg_time:.2f} ms")
        print(f"  吞吐量: {fps:.2f} FPS")
    
    # 保存結果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n結果已保存到 {output_file}")
    
    # 顯示比較表格
    print("\n=== TensorRT批處理大小性能 ===")
    print(f"{'批處理大小':<10} | {'平均時間 (ms)':<15} | {'吞吐量 (FPS)':<15}")
    print("-" * 50)
    
    for batch_size in sorted(results.keys()):
        data = results[batch_size]
        print(f"{batch_size:<10} | {data['avg_time_ms']:<15.2f} | {data['fps']:<15.2f}")
    
    return results

if __name__ == "__main__":
    run_batch_tests() 