from tensorflow.keras.datasets import boston_housing

# 加载数据
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# 数据标准化（重要步骤）
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

from tensorflow.keras import models
from tensorflow.keras import layers

def build_model():
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  # 输出层不需要激活函数
    ])
    return model

model = build_model()
model.compile(optimizer='rmsprop',
              loss='mse',  # 均方误差
              metrics=['mae'])  # 平均绝对误差


history = model.fit(train_data, train_targets,
                    epochs=100,
                    batch_size=16,
                    validation_split=0.2)

# 在测试集上评估
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(f"测试集MAE: {test_mae_score}")

# 对新数据进行预测
sample = test_data[0]  # 取测试集第一个样本
prediction = model.predict(sample.reshape(1, -1))
print(f"预测价格: {prediction[0][0]}, 实际价格: {test_targets[0]}")
