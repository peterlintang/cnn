
import tensorflow as tf

# 创建并训练一个简单模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

# 保存为SavedModel格式
model.save('my_model')  # 注意：没有文件扩展名


# 保存为HDF5格式
model.save('my_model.h5')  # 注意.h5扩展名


# 从SavedModel加载
loaded_model = tf.keras.models.load_model('my_model')

# 验证模型
loss, acc = loaded_model.evaluate(x_test, y_test, verbose=2)
print(f"Restored model, accuracy: {100*acc:.1f}%")

# 从HDF5文件加载
loaded_model = tf.keras.models.load_model('my_model.h5')

# 验证模型
loss, acc = loaded_model.evaluate(x_test, y_test, verbose=2)
print(f"Restored model, accuracy: {100*acc:.1f}%")

# 保存权重
model.save_weights('my_model_weights')

# 保存为HDF5格式的权重
model.save_weights('my_model_weights.h5')

# 创建相同架构的模型
new_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
new_model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

# 加载权重
new_model.load_weights('my_model_weights')

# 或者对于.h5文件
new_model.load_weights('my_model_weights.h5')

# 创建检查点回调
checkpoint_path = "training_1/cp.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1)

# 使用回调训练模型
model.fit(x_train, y_train,
          epochs=10,
          callbacks=[cp_callback])

