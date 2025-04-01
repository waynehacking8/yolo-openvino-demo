import os
import time
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

def load_images(image_dir, max_images=None):
    """載入圖片"""
    images = []
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    if max_images:
        image_files = image_files[:max_images]
    
    for filename in image_files:
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        if img is not None:
            # 轉換為RGB並調整大小
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 640))
            # 將圖片轉換為PyTorch張量並正規化
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            images.append(img_tensor)
    
    return images

def run_batch_inference(model_path, images, batch_sizes=[1, 2, 4, 8], runs=5, warmup=2):
    """執行批次推理並測量性能"""
    results = {}
    
    # 檢查是否有CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("警告：未檢測到CUDA，將使用CPU進行測試")
    
    # 加載模型
    model = torch.jit.load(model_path)
    model.to(device)
    model.eval()
    
    # 對每個批次大小進行測試
    for batch_size in batch_sizes:
        print(f"測試批次大小: {batch_size}")
        results[batch_size] = {}
        
        # 準備批次
        batches = []
        for i in range(0, len(images), batch_size):
            batch = images[i:min(i+batch_size, len(images))]
            # 如果最後一個批次不足，填充
            if len(batch) < batch_size:
                batch = batch + [images[0]] * (batch_size - len(batch))
            batch_tensor = torch.stack(batch)
            batches.append(batch_tensor)
        
        print(f"  創建了 {len(batches)} 個批次")
        
        # 預熱
        print("  預熱中...")
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(batches[0].to(device))
        
        # 測試
        print("  進行推理測試...")
        times = []
        with torch.no_grad():
            for _ in range(runs):
                for batch in batches:
                    batch = batch.to(device)
                    
                    torch.cuda.synchronize()
                    start_time = time.time()
                    _ = model(batch)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    
                    times.append((end_time - start_time) * 1000)  # 轉換為毫秒
        
        # 計算統計資料
        avg_time = np.mean(times)
        fps = 1000 / avg_time * batch_size  # 調整為每秒處理的圖片數
        
        results[batch_size] = {
            "avg_time_ms": float(avg_time),
            "fps": float(fps),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "std_ms": float(np.std(times))
        }
        
        print(f"  平均批次處理時間: {avg_time:.2f} ms")
        print(f"  吞吐量: {fps:.2f} FPS")
    
    return results

def plot_results(results):
    """繪製結果圖表"""
    batch_sizes = sorted(results.keys())
    times = [results[batch]["avg_time_ms"] for batch in batch_sizes]
    fps = [results[batch]["fps"] for batch in batch_sizes]
    
    # 設置繪圖風格
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 10))
    
    # 繪製處理時間圖表
    plt.subplot(2, 1, 1)
    plt.bar(range(len(batch_sizes)), times, color='skyblue')
    plt.xticks(range(len(batch_sizes)), [str(b) for b in batch_sizes])
    plt.xlabel('批次大小')
    plt.ylabel('處理時間 (ms)')
    plt.title('不同批次大小的處理時間 (PyTorch CUDA)')
    
    # 繪製吞吐量圖表
    plt.subplot(2, 1, 2)
    plt.bar(range(len(batch_sizes)), fps, color='green')
    plt.xticks(range(len(batch_sizes)), [str(b) for b in batch_sizes])
    plt.xlabel('批次大小')
    plt.ylabel('吞吐量 (FPS)')
    plt.title('不同批次大小的吞吐量 (PyTorch CUDA)')
    
    plt.tight_layout()
    plt.savefig('benchmark_results/cuda_batch_performance.png', dpi=200)
    print("圖表已保存至 benchmark_results/cuda_batch_performance.png")

def main():
    """主函數"""
    model_path = "yolov8n.torchscript"
    image_dir = "test_images"
    output_file = "benchmark_results/pytorch_cuda_batch_results.json"
    batch_sizes = [1, 2, 4, 8]
    
    print("載入圖片...")
    images = load_images(image_dir)
    print(f"已載入 {len(images)} 張圖片")
    
    print("開始批次推理測試...")
    results = run_batch_inference(model_path, images, batch_sizes=batch_sizes)
    
    # 保存結果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"結果已保存至 {output_file}")
    
    # 繪製圖表
    plot_results(results)
    
    # 輸出比較表格
    print("\n=== PyTorch CUDA 批次處理性能 ===")
    print(f"{'批次大小':<10} | {'平均時間 (ms)':<15} | {'吞吐量 (FPS)':<15}")
    print("-" * 50)
    
    for batch_size in sorted(results.keys()):
        data = results[batch_size]
        print(f"{batch_size:<10} | {data['avg_time_ms']:<15.2f} | {data['fps']:<15.2f}")

if __name__ == "__main__":
    main()
