
# 完整的训练步骤示例
import tensorflow as tf

# 模型和数据
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])

x_train = tf.random.normal([100, 5])
y_train = tf.random.normal([100, 1])

optimizer = tf.keras.optimizers.Adam(0.01)

# 训练步骤
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = tf.keras.losses.mse(y, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# 执行训练
for epoch in range(10):
    loss = train_step(x_train, y_train)
#    print(loss)
    print(f"Epoch {epoch}: Loss = {loss[0]:.4f}")
