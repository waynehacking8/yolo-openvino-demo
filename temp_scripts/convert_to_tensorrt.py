import os
import tensorrt as trt
import numpy as np
from pathlib import Path

def build_engine(onnx_path, engine_path, batch_size=1):
    """Build TensorRT engine from ONNX model"""
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    
    # 設置最大內存和性能配置
    config.max_workspace_size = 4 * 1 << 30  # 4GB
    config.set_flag(trt.BuilderFlag.FP16)
    
    # 解析ONNX模型
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"ONNX解析錯誤: {parser.get_error(error)}")
            return False
    
    # 設置批處理大小
    print(f"正在創建支持多個批處理大小的引擎...")
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    input_shape = network.get_input(0).shape
    
    # 設置動態批處理大小範圍 (1-16)
    min_shape = (1,) + tuple(input_shape[1:])
    opt_shape = (8,) + tuple(input_shape[1:]) 
    max_shape = (16,) + tuple(input_shape[1:])
    
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    
    # 序列化引擎
    print(f"正在構建TensorRT引擎: {engine_path}")
    engine = builder.build_engine(network, config)
    if engine:
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        print(f"TensorRT引擎已保存到: {engine_path}")
        return True
    else:
        print("引擎構建失敗")
        return False

if __name__ == "__main__":
    onnx_path = "yolov8n.onnx"
    engine_path = "yolov8n.engine"
    
    print(f"TensorRT版本: {trt.__version__}")
    success = build_engine(onnx_path, engine_path)
    
    if success:
        print("成功建立TensorRT引擎，可用於批處理測試")
    else:
        print("建立TensorRT引擎失敗") 