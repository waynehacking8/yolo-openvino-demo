import os
import json
import matplotlib.pyplot as plt
import numpy as np

def combine_results():
    """結合TensorRT和CPU後端的結果"""
    # 載入TensorRT結果
    tensorrt_file = "benchmark_results/tensorrt_batch_results.json"
    if not os.path.exists(tensorrt_file):
        print(f"錯誤: {tensorrt_file} 不存在")
        return
    
    with open(tensorrt_file, 'r') as f:
        tensorrt_results = json.load(f)
    
    # 載入CPU後端結果
    cpu_file = "benchmark_results/batch_inference_results.json"
    if not os.path.exists(cpu_file):
        print(f"錯誤: {cpu_file} 不存在")
        return
    
    with open(cpu_file, 'r') as f:
        cpu_results = json.load(f)
    
    # 合併結果
    combined_results = {}
    for batch_size in cpu_results:
        combined_results[batch_size] = cpu_results[batch_size]
        if batch_size in tensorrt_results:
            combined_results[batch_size]["tensorrt_gpu"] = tensorrt_results[batch_size]
    
    # 保存合併的結果
    output_file = "benchmark_results/combined_batch_results.json"
    with open(output_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"合併結果已保存到 {output_file}")
    
    # 繪製圖表
    plot_combined_results(combined_results)
    
    return combined_results

def plot_combined_results(results):
    """繪製所有後端的比較圖表"""
    batch_sizes = sorted([int(k) for k in results.keys()])
    backends = ["pytorch_cpu", "openvino_cpu", "tensorrt_gpu"]
    backend_colors = {
        "pytorch_cpu": "#1f77b4",    # 藍色
        "openvino_cpu": "#ff7f0e",   # 橙色
        "tensorrt_gpu": "#2ca02c"    # 綠色
    }
    
    # 初始化數據結構
    batch_times = {backend: [] for backend in backends}
    throughputs = {backend: [] for backend in backends}
    
    # 提取數據
    for batch_size in batch_sizes:
        batch_size_str = str(batch_size)
        for backend in backends:
            if backend in results[batch_size_str]:
                batch_times[backend].append(results[batch_size_str][backend]["avg_time_ms"])
                throughputs[backend].append(results[batch_size_str][backend]["fps"])
            else:
                # 如果後端數據缺失，添加None以保持對齊
                batch_times[backend].append(None)
                throughputs[backend].append(None)
    
    # 設置圖表風格
    plt.style.use('ggplot')
    plt.figure(figsize=(14, 10))
    
    # 繪製批處理時間圖表
    plt.subplot(2, 1, 1)
    bar_width = 0.25
    index = np.arange(len(batch_sizes))
    
    for i, backend in enumerate(backends):
        # 過濾None值
        valid_indices = [j for j, val in enumerate(batch_times[backend]) if val is not None]
        valid_times = [batch_times[backend][j] for j in valid_indices]
        valid_x = [index[j] + i*bar_width for j in valid_indices]
        
        if valid_times:
            plt.bar(valid_x, valid_times, bar_width,
                    label=backend, alpha=0.8, color=backend_colors[backend])
    
    plt.xlabel('批處理大小')
    plt.ylabel('處理時間 (ms)')
    plt.title('各後端批處理時間比較')
    plt.xticks(index + bar_width, batch_sizes)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 繪製吞吐量圖表
    plt.subplot(2, 1, 2)
    
    for i, backend in enumerate(backends):
        # 過濾None值
        valid_indices = [j for j, val in enumerate(throughputs[backend]) if val is not None]
        valid_fps = [throughputs[backend][j] for j in valid_indices]
        valid_x = [index[j] + i*bar_width for j in valid_indices]
        
        if valid_fps:
            plt.bar(valid_x, valid_fps, bar_width,
                    label=backend, alpha=0.8, color=backend_colors[backend])
    
    plt.xlabel('批處理大小')
    plt.ylabel('吞吐量 (FPS)')
    plt.title('各後端批處理吞吐量比較')
    plt.xticks(index + bar_width, batch_sizes)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('benchmark_results/batch_processing_all_backends.png', dpi=200)
    print("批處理性能比較圖表已保存到 benchmark_results/batch_processing_all_backends.png")
    
    # 創建速度比率圖
    plt.figure(figsize=(10, 6))
    
    # 對於每個批處理大小，計算相對於TensorRT的速度比率
    for i, batch_size in enumerate(batch_sizes):
        batch_size_str = str(batch_size)
        
        if "tensorrt_gpu" in results[batch_size_str]:
            trt_time = results[batch_size_str]["tensorrt_gpu"]["avg_time_ms"]
            
            for j, backend in enumerate(["pytorch_cpu", "openvino_cpu"]):
                if backend in results[batch_size_str]:
                    backend_time = results[batch_size_str][backend]["avg_time_ms"]
                    speed_ratio = backend_time / trt_time
                    
                    # 繪製柱狀圖
                    plt.bar(i + (0 if backend == "pytorch_cpu" else 0.35), 
                            speed_ratio, 0.35, alpha=0.8,
                            color=backend_colors[backend],
                            label=backend if i == 0 else "")
                    
                    # 添加文字標籤
                    plt.text(i + (0 if backend == "pytorch_cpu" else 0.35),
                             speed_ratio + 0.2,
                             f'{speed_ratio:.2f}x',
                             ha='center')
    
    plt.xlabel('批處理大小')
    plt.ylabel('相對於TensorRT的速度比率')
    plt.title('CPU後端相對於TensorRT的速度比較')
    plt.xticks(np.arange(len(batch_sizes)) + 0.175, batch_sizes)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('benchmark_results/speed_ratio_vs_tensorrt.png', dpi=200)
    print("速度比率圖表已保存到 benchmark_results/speed_ratio_vs_tensorrt.png")

if __name__ == "__main__":
    combined_results = combine_results()
    
    # 如果沒有合併結果，則嘗試只使用TensorRT結果
    if not combined_results:
        tensorrt_file = "benchmark_results/tensorrt_batch_results.json"
        if os.path.exists(tensorrt_file):
            with open(tensorrt_file, 'r') as f:
                tensorrt_results = json.load(f)
            
            # 轉換為字符串鍵
            string_key_results = {str(k): v for k, v in tensorrt_results.items()}
            
            # 繪製TensorRT結果
            print("僅繪製TensorRT結果...")
            
            # 繪製批處理時間和吞吐量圖表
            batch_sizes = sorted([int(k) for k in string_key_results.keys()])
            times = [string_key_results[str(k)]["avg_time_ms"] for k in batch_sizes]
            fps = [string_key_results[str(k)]["fps"] for k in batch_sizes]
            
            plt.figure(figsize=(12, 8))
            
            # 批處理時間
            plt.subplot(2, 1, 1)
            plt.bar(range(len(batch_sizes)), times, alpha=0.8, color="#2ca02c")
            plt.xlabel('批處理大小')
            plt.ylabel('處理時間 (ms)')
            plt.title('TensorRT批處理時間')
            plt.xticks(range(len(batch_sizes)), batch_sizes)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 吞吐量
            plt.subplot(2, 1, 2)
            plt.bar(range(len(batch_sizes)), fps, alpha=0.8, color="#2ca02c")
            plt.xlabel('批處理大小')
            plt.ylabel('吞吐量 (FPS)')
            plt.title('TensorRT批處理吞吐量')
            plt.xticks(range(len(batch_sizes)), batch_sizes)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig('benchmark_results/tensorrt_batch_performance.png', dpi=200)
            print("TensorRT性能圖表已保存到 benchmark_results/tensorrt_batch_performance.png") 