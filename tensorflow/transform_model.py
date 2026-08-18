import tensorflow as tf

# 保存为 SavedModel
model.save('my_model', save_format='tf')

# 加载 SavedModel
loaded_model = tf.keras.models.load_model('my_model')


# 转换模型为 TFLite 格式
converter = tf.lite.TFLiteConverter.from_saved_model('my_model')
tflite_model = converter.convert()

# 保存转换后的模型
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# 动态范围量化（最简单的量化方式）
converter = tf.lite.TFLiteConverter.from_saved_model('my_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

import tensorflow_model_optimization as tfmot

# 定义剪枝参数
prune_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.50,
        final_sparsity=0.90,
        begin_step=0,
        end_step=1000
    )
}

# 应用剪枝
model = tf.keras.Sequential([...])  # 你的模型
model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **prune_params)

# 训练剪枝模型
model_for_pruning.compile(...)
model_for_pruning.fit(...)

# 去除剪枝包装
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

# 使用 TF-TRT 转换器
from tensorflow.python.compiler.tensorrt import trt_convert as trt

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir='my_model',
    precision_mode=trt.TrtPrecisionMode.FP16
)
converter.convert()
converter.save('trt_optimized_model')

import coremltools as ct

# 从 SavedModel 转换
mlmodel = ct.convert('my_model')

# 保存 Core ML 模型
mlmodel.save('model.mlmodel')
