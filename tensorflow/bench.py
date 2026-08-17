import tensorflow as tf
import time

print("TensorFlow Performance Test")
print("-" * 30)

# 矩阵乘法测试
def benchmark_matmul(device_name, size=1000):
    with tf.device(device_name):
        a = tf.random.normal([size, size])
        b = tf.random.normal([size, size])
        
        # 预热
        for _ in range(5):
            c = tf.matmul(a, b)
        
        # 计时
        start_time = time.time()
        for _ in range(10):
            c = tf.matmul(a, b)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        return avg_time

# CPU 测试
cpu_time = benchmark_matmul('/CPU:0')
print(f"CPU ({1000}x{1000} matmul): {cpu_time:.4f} seconds")

# GPU 测试（如果可用）
if tf.config.list_physical_devices('GPU'):
    gpu_time = benchmark_matmul('/GPU:0')
    print(f"GPU ({1000}x{1000} matmul): {gpu_time:.4f} seconds")
    print(f"GPU speedup: {cpu_time/gpu_time:.2f}x")
else:
    print("No GPU available for testing")
