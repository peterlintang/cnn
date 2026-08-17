import tensorflow as tf
import numpy as np

# 从Python列表创建
tensor_from_list = tf.constant([[1, 2], [3, 4]])
print(tensor_from_list)

# 从NumPy数组创建
numpy_array = np.array([[5, 6], [7, 8]])
tensor_from_numpy = tf.constant(numpy_array)
print(tensor_from_numpy)
