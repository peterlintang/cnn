import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, models

# 加载数据
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# 类别名称
class_names = ['飞机', '汽车', '鸟', '猫', '鹿', 
               '狗', '青蛙', '马', '船', '卡车']

# 归一化像素值到0-1范围
train_images = train_images / 255.0
test_images = test_images / 255.0

# 查看数据形状
print("训练集图像形状:", train_images.shape)
print("训练集标签形状:", train_labels.shape)


model = models.Sequential([
    # 卷积层1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    # 卷积层2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # 卷积层3
    layers.Conv2D(64, (3, 3), activation='relu'),

    # 全连接层
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)  # 输出层，10个类别
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\n测试准确率: {test_acc}')

import numpy as np

# 添加softmax层使输出为概率
probability_model = tf.keras.Sequential([model, layers.Softmax()])

# 对测试集前5张图进行预测
predictions = probability_model.predict(test_images[:5])

# 显示预测结果
for i in range(5):
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i][0]
    print(f"预测: {class_names[predicted_label]} | 实际: {class_names[true_label]}")

