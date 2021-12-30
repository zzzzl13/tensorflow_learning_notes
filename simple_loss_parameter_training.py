import tensorflow as tf
import numpy as np

w = tf.Variable(tf.constant(5,dtype=tf.float32))    # variable 对象声明  此对象下才能对目标进行训练更新
lr = 0.2   # 学习率  可调整为0.001  0.99等观察梯度下降后损失函数的结果
e_time = 40

for i in range(e_time):
    with tf.GradientTape() as tape:
        loss = tf.square(w + 1)     # loss = (w+1)^2
        grads = tape.gradient(loss,w)   #   loss对w求导
    
    w.assign_sub(lr*grads)   #  对参数进行自减  即 w -= lr*grads   即训练更新
    print('经过%s次计算， w为%s，loss为%s'%(i,w.numpy(),loss.numpy()))