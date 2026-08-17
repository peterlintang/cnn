#!/usr/bin/env python3
"""
TensorFlow 安装验证脚本
"""

import sys
print("Python version:", sys.version)
print("-" * 50)

# 检查 TensorFlow
try:
    import tensorflow as tf
    print(f"✓ TensorFlow version: {tf.__version__}")
    print(f"✓ Keras version: {tf.keras.__version__}")
    
    # 检查 GPU 支持
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"✓ GPU devices found: {len(physical_devices)}")
        for i, device in enumerate(physical_devices):
            print(f"  - GPU {i}: {device}")
        print(f"✓ CUDA built: {tf.test.is_built_with_cuda()}")
    else:
        print("&#x26a0; No GPU devices found (CPU only)")
    
    # 简单计算测试
    a = tf.constant([1, 2, 3])
    b = tf.constant([4, 5, 6])
    c = tf.add(a, b)
    print(f"✓ Basic computation test: {a.numpy()} + {b.numpy()} = {c.numpy()}")
    
except ImportError as e:
    print(f"✗ TensorFlow import failed: {e}")

print("-" * 50)

# 检查其他重要包
packages = ['numpy', 'pandas', 'matplotlib', 'sklearn', 'jupyter']
for package in packages:
    try:
        module = __import__(package)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package}: {version}")
    except ImportError:
        print(f"✗ {package}: not installed")

print("-" * 50)

# 内存和设备信息
print("System Information:")
if tf.config.list_physical_devices('GPU'):
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU memory growth enabled for {gpu}")
        except:
            print(f"&#x26a0; Could not set memory growth for {gpu}")

print("Installation verification complete!")
