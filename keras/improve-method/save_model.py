'''
断点续训 存储模型
将训练后的模型参数存储起来 便于有新的数据能够进行续训 优化模型
'''
import tensorflow as tf
import os

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/mnist.ckpt"    # 定义存储文件名
if os.path.exists(checkpoint_save_path + '.index'):  #  模型存储后会同步生成.index索引文件  因此可以此判断此前是否有模型被存储
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)     # 如果存在 读取模型

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)   # 定义模型存储回调函数 两个True表示存储模型参数并且只存最优值

history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])  # 添加模型存储回调函数
model.summary()
