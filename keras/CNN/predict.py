'''
模型复现 加载参数
使用预测组中图片导入网络进行预测 并于实际值对比
'''
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout


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

chekoutpoint_path = 'D:/python scripts/learn!/training backup/checkpoint/cifar10.ckpt'

model.load_weights(chekoutpoint_path)  

label =['飞机','汽车','鸟','猫','鹿','狗','青蛙','马','船','卡车']  # cifar10图片标签
for i in range(6,30):
    print('实际值：'+ label[y_test[i][0]])
    pic = x_test[i][tf.newaxis,...]    # 转换数据维度 使之能够传入神经网络  原维度(32,32,3)->现维度(1,32,32,3)
    b = model.predict(pic)
    mlab = tf.argmax(b,axis=1)
    print('预测值:' + label[mlab.numpy()[0]])




