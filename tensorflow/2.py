# 示例张量
import tensorflow as tf

# 创建一个 2x3 的矩阵张量
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])

print(f"形状 (Shape): {tensor.shape}")        # (2, 3)
print(f"数据类型 (Dtype): {tensor.dtype}")    # int32
print(f"维度 (Rank): {tf.rank(tensor)}")      # 2 
print(f"设备 (Device): {tensor.device}")      # /job:localhost/replica:0/task:0/device:CPU:0


# 简单的前向传播示例

# 输入数据
x = tf.constant([[1.0, 2.0]])

# 模型参数
W1 = tf.Variable(tf.random.normal([2, 3]))
b1 = tf.Variable(tf.zeros([3]))
W2 = tf.Variable(tf.random.normal([3, 1]))
b2 = tf.Variable(tf.zeros([1]))

# 前向传播
hidden = tf.nn.relu(tf.matmul(x, W1) + b1)  # 隐层
output = tf.matmul(hidden, W2) + b2         # 输出层

print(f"最终输出: {output}")

# 自动微分示例
x = tf.Variable(3.0)

# 使用 GradientTape 记录运算
with tf.GradientTape() as tape:
    y = x**2 + 2*x + 1  # y = x² + 2x + 1

# 计算 dy/dx
gradient = tape.gradient(y, x)
print(f"当 x=3 时，dy/dx = {gradient}")  # 应该是 2x + 2 = 8
