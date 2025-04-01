import os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_benchmark_data():
    """Load benchmark data from JSON file"""
    benchmark_file = "benchmark_results/benchmark_summary_20250401_162759.json"
    if not os.path.exists(benchmark_file):
        print(f"Error: {benchmark_file} does not exist")
        return None
    
    with open(benchmark_file, 'r') as f:
        data = json.load(f)
    
    # Extract backends data
    benchmark_data = data["backends"]
    
    return benchmark_data

def plot_benchmark_comparison(benchmark_data):
    """Create benchmark comparison visualization in English"""
    if not benchmark_data:
        return
    
    backends = list(benchmark_data.keys())
    # Sort backends with TensorRT first
    backends.sort(key=lambda x: 0 if "tensorrt" in x else 1)
    
    backend_colors = {
        "pytorch_cpu": "#1f77b4",    # Blue
        "openvino_cpu": "#ff7f0e",   # Orange
        "tensorrt_gpu": "#2ca02c"    # Green
    }
    
    # Extract data
    inf_times = [benchmark_data[backend]["inference_time"] for backend in backends]
    fps_values = [benchmark_data[backend]["fps"] for backend in backends]
    
    # Calculate speed ratio compared to fastest backend
    min_time = min(inf_times)
    speed_ratios = [time / min_time for time in inf_times]
    
    # Set chart style
    plt.style.use('ggplot')
    plt.figure(figsize=(14, 14))
    
    # Plot inference time
    plt.subplot(3, 1, 1)
    bars = plt.bar(backends, inf_times, alpha=0.8, 
           color=[backend_colors.get(backend, "#d62728") for backend in backends])
    
    # Add value labels
    for bar, value in zip(bars, inf_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.2f} ms', ha='center')
    
    plt.ylabel('Inference Time (ms)')
    plt.title('Single Image Inference Time Comparison')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot FPS
    plt.subplot(3, 1, 2)
    bars = plt.bar(backends, fps_values, alpha=0.8,
           color=[backend_colors.get(backend, "#d62728") for backend in backends])
    
    # Add value labels
    for bar, value in zip(bars, fps_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.2f} FPS', ha='center')
    
    plt.ylabel('Throughput (FPS)')
    plt.title('Single Image Inference Throughput Comparison')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot speed ratio
    plt.subplot(3, 1, 3)
    bars = plt.bar(backends, speed_ratios, alpha=0.8,
           color=[backend_colors.get(backend, "#d62728") for backend in backends])
    
    # Add value labels 
    for bar, value, backend in zip(bars, speed_ratios, backends):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}x', ha='center')
    
    plt.ylabel('Speed Ratio')
    plt.title('Performance Comparison Relative to Fastest Backend')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save figure with specified filename
    plt.savefig('benchmark_results/benchmark_comparison_20250401_162759.png', dpi=200)
    print(f"Benchmark comparison chart saved to benchmark_results/benchmark_comparison_20250401_162759.png")
    
    # Also save with fixed name for README reference
    plt.savefig('benchmark_results/benchmark_comparison.png', dpi=200)
    print(f"Benchmark comparison chart also saved to benchmark_results/benchmark_comparison.png")
    
if __name__ == "__main__":
    benchmark_data = load_benchmark_data()
    if benchmark_data:
        plot_benchmark_comparison(benchmark_data)
    else:
        print("Failed to load benchmark data") 