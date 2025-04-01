import os
import json
import matplotlib.pyplot as plt
import numpy as np

def combine_results():
    """Combine TensorRT and CPU backend results"""
    # Load TensorRT results
    tensorrt_file = "benchmark_results/tensorrt_batch_results.json"
    if not os.path.exists(tensorrt_file):
        print(f"Error: {tensorrt_file} does not exist")
        return
    
    with open(tensorrt_file, 'r') as f:
        tensorrt_results = json.load(f)
    
    # Load CPU backend results
    cpu_file = "benchmark_results/batch_inference_results.json"
    if not os.path.exists(cpu_file):
        print(f"Error: {cpu_file} does not exist")
        return
    
    with open(cpu_file, 'r') as f:
        cpu_results = json.load(f)
    
    # Merge results
    combined_results = {}
    for batch_size in cpu_results:
        combined_results[batch_size] = cpu_results[batch_size]
        if batch_size in tensorrt_results:
            combined_results[batch_size]["tensorrt_gpu"] = tensorrt_results[batch_size]
    
    # Save combined results
    output_file = "benchmark_results/combined_batch_results.json"
    with open(output_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"Combined results saved to {output_file}")
    
    # Plot charts
    plot_combined_results(combined_results)
    
    return combined_results

def plot_combined_results(results):
    """Plot comparison charts for all backends"""
    batch_sizes = sorted([int(k) for k in results.keys()])
    backends = ["pytorch_cpu", "openvino_cpu", "tensorrt_gpu"]
    backend_colors = {
        "pytorch_cpu": "#1f77b4",    # Blue
        "openvino_cpu": "#ff7f0e",   # Orange
        "tensorrt_gpu": "#2ca02c"    # Green
    }
    
    # Initialize data structures
    batch_times = {backend: [] for backend in backends}
    throughputs = {backend: [] for backend in backends}
    
    # Extract data
    for batch_size in batch_sizes:
        batch_size_str = str(batch_size)
        for backend in backends:
            if backend in results[batch_size_str]:
                batch_times[backend].append(results[batch_size_str][backend]["avg_time_ms"])
                throughputs[backend].append(results[batch_size_str][backend]["fps"])
            else:
                # Add None for missing backend data to maintain alignment
                batch_times[backend].append(None)
                throughputs[backend].append(None)
    
    # Set chart style
    plt.style.use('ggplot')
    plt.figure(figsize=(14, 10))
    
    # Plot batch processing time chart
    plt.subplot(2, 1, 1)
    bar_width = 0.25
    index = np.arange(len(batch_sizes))
    
    for i, backend in enumerate(backends):
        # Filter None values
        valid_indices = [j for j, val in enumerate(batch_times[backend]) if val is not None]
        valid_times = [batch_times[backend][j] for j in valid_indices]
        valid_x = [index[j] + i*bar_width for j in valid_indices]
        
        if valid_times:
            plt.bar(valid_x, valid_times, bar_width,
                    label=backend, alpha=0.8, color=backend_colors[backend])
    
    plt.xlabel('Batch Size')
    plt.ylabel('Processing Time (ms)')
    plt.title('Batch Processing Time Comparison Across Backends')
    plt.xticks(index + bar_width, batch_sizes)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot throughput chart
    plt.subplot(2, 1, 2)
    
    for i, backend in enumerate(backends):
        # Filter None values
        valid_indices = [j for j, val in enumerate(throughputs[backend]) if val is not None]
        valid_fps = [throughputs[backend][j] for j in valid_indices]
        valid_x = [index[j] + i*bar_width for j in valid_indices]
        
        if valid_fps:
            plt.bar(valid_x, valid_fps, bar_width,
                    label=backend, alpha=0.8, color=backend_colors[backend])
    
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (FPS)')
    plt.title('Batch Processing Throughput Comparison Across Backends')
    plt.xticks(index + bar_width, batch_sizes)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('benchmark_results/batch_processing_all_backends.png', dpi=200)
    print("Batch processing performance comparison chart saved to benchmark_results/batch_processing_all_backends.png")
    
    # Create speed ratio chart
    plt.figure(figsize=(10, 6))
    
    # Calculate speed ratio relative to TensorRT for each batch size
    for i, batch_size in enumerate(batch_sizes):
        batch_size_str = str(batch_size)
        
        if "tensorrt_gpu" in results[batch_size_str]:
            trt_time = results[batch_size_str]["tensorrt_gpu"]["avg_time_ms"]
            
            for j, backend in enumerate(["pytorch_cpu", "openvino_cpu"]):
                if backend in results[batch_size_str]:
                    backend_time = results[batch_size_str][backend]["avg_time_ms"]
                    speed_ratio = backend_time / trt_time
                    
                    # Plot bar chart
                    plt.bar(i + (0 if backend == "pytorch_cpu" else 0.35), 
                            speed_ratio, 0.35, alpha=0.8,
                            color=backend_colors[backend],
                            label=backend if i == 0 else "")
                    
                    # Add text labels
                    plt.text(i + (0 if backend == "pytorch_cpu" else 0.35),
                             speed_ratio + 0.2,
                             f'{speed_ratio:.2f}x',
                             ha='center')
    
    plt.xlabel('Batch Size')
    plt.ylabel('Speed Ratio Relative to TensorRT')
    plt.title('CPU Backends Speed Comparison Relative to TensorRT')
    plt.xticks(np.arange(len(batch_sizes)) + 0.175, batch_sizes)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('benchmark_results/speed_ratio_vs_tensorrt.png', dpi=200)
    print("Speed ratio chart saved to benchmark_results/speed_ratio_vs_tensorrt.png")

def plot_tensorrt_only(tensorrt_results):
    """Plot TensorRT-only performance charts"""
    # Convert to string keys if needed
    string_key_results = {str(k): v for k, v in tensorrt_results.items()}
    
    # Plot TensorRT results
    print("Plotting TensorRT results only...")
    
    # Plot batch processing time and throughput charts
    batch_sizes = sorted([int(k) for k in string_key_results.keys()])
    times = [string_key_results[str(k)]["avg_time_ms"] for k in batch_sizes]
    fps = [string_key_results[str(k)]["fps"] for k in batch_sizes]
    
    plt.figure(figsize=(12, 8))
    
    # Batch processing time
    plt.subplot(2, 1, 1)
    plt.bar(range(len(batch_sizes)), times, alpha=0.8, color="#2ca02c")
    plt.xlabel('Batch Size')
    plt.ylabel('Processing Time (ms)')
    plt.title('TensorRT Batch Processing Time')
    plt.xticks(range(len(batch_sizes)), batch_sizes)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Throughput
    plt.subplot(2, 1, 2)
    plt.bar(range(len(batch_sizes)), fps, alpha=0.8, color="#2ca02c")
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (FPS)')
    plt.title('TensorRT Batch Processing Throughput')
    plt.xticks(range(len(batch_sizes)), batch_sizes)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('benchmark_results/tensorrt_batch_performance.png', dpi=200)
    print("TensorRT performance charts saved to benchmark_results/tensorrt_batch_performance.png")

if __name__ == "__main__":
    combined_results = combine_results()
    
    # If no combined results, try using only TensorRT results
    if not combined_results:
        tensorrt_file = "benchmark_results/tensorrt_batch_results.json"
        if os.path.exists(tensorrt_file):
            with open(tensorrt_file, 'r') as f:
                tensorrt_results = json.load(f)
            
            # Plot TensorRT results
            plot_tensorrt_only(tensorrt_results) 