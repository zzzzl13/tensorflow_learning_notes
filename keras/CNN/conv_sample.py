import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout
from tensorflow.keras import Model
from matplotlib import pyplot as plt
from PIL import Image
import os

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.Sequential([
    Conv2D(filters=6, kernel_size=(5,5), padding='same'),  # 卷积层  深度为6  卷积核大小为（5x5） 全零填充
    BatchNormalization(),    # 批标准化
    Activation('relu'),      # relu激活函数
    MaxPool2D(pool_size = (2,2), strides = 2, padding = 'same'),   # 最大值池化  全零填充
    Flatten(),       # 拉直
    Dense(128, activation = 'relu'),     # 全连接层
    Dropout(0.2),                        # dropout
    Dense(10, activation = 'softmax')    # 全连接层
])


model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),  #优化器
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  #损失函数
              metrics=['sparse_categorical_accuracy'])   #准确率

chekoutpoint_path = 'D:/python scripts/learn!/training backup/checkpoint/cifar10.ckpt'
if os.path.exists(chekoutpoint_path + '.index'):  #  模型存储后会同步生成.index索引文件  因此可以此判断此前是否有模型被存储
    print('-------------load the model-----------------')
    model.load_weights(chekoutpoint_path)     # 如果存在 读取模型
    
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=chekoutpoint_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)   # 定义模型存储回调函数 两个True表示存储模型参数并且只存最优值

result = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1, callbacks = [cp_callback])
model.summary()

'''-------准确率 损失函数结果---------'''
acc = result.history['sparse_categorical_accuracy']
val_acc = result.history['val_sparse_categorical_accuracy']
loss = result.history['loss']
val_loss = result.history['val_loss']
plt.subplot(1,2,1)
plt.plot(acc,label='Training Accuracy')
plt.plot(val_acc,label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(loss,label='Training Loss')
plt.plot(val_loss,label='Validation Loss')
plt.title('Loss')
plt.legend()
plt.show()
    



