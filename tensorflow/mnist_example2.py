import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# 1. 加载数据
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. 预处理
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# 3. 构建模型
model = keras.Sequential([
    layers.Dense(512, activation="relu", input_shape=(784,)),
    layers.Dense(10, activation="softmax")
])

#model.add(layers.Dropout(0.5))
#model.compile(optimizer="adam", ...)
#model.add(layers.Dense(256, activation="relu"))
#model.add(layers.Conv2D(32, (3, 3), activation="relu"))

# 4. 编译模型
model.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 5. 训练模型
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.2
)

# 6. 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"测试准确率: {test_acc:.4f}")
