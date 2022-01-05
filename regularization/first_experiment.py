import csv
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
x_data = []
y_data = []
with open("dot.csv",'r') as f:
    csvreader = csv.reader(f)
    for row in csvreader:
        for i in range(len(row)):
            try:
                row[i] = float(row[i])
            except:
                pass
        x_data.append(row[0:2])
        y_data.append(row[2:])
del x_data[0]
del y_data[0]
y_c = ['red' if each == [0] else 'blue' for each in y_data]
plt.scatter(np.array(x_data)[:,0],np.array(x_data)[:,1],color=np.array(y_c))
x_data = tf.constant(x_data)
y_data = tf.constant(y_data)
x_data = tf.cast(x_data,tf.float32)
y_data = tf.cast(y_data,tf.float32)
train_db = tf.data.Dataset.from_tensor_slices((x_data,y_data)).batch(50)

w1 = tf.Variable(tf.random.truncated_normal([2, 8],stddev=0.1),dtype = tf.float32)    #开始时设置时尝试使用输入层-输出层结构  结果无法收敛 后尝试增加一层隐藏层并增加神经元
w2 = tf.Variable(tf.random.truncated_normal([8, 1],stddev=0.1),dtype = tf.float32)    # 此处为输入层（2个神经元）-隐藏层（8个神经元）-输出层（1个神经元）
b1 = tf.Variable(tf.random.truncated_normal([8],stddev=0.1),dtype = tf.float32)
b2 = tf.Variable(tf.random.truncated_normal([1]),dtype = tf.float32)


lr = 0.01   # 可根据loss修改 寻找较合适的值
epochs = 800  
loss_all = 0
train_loss_results = []
for epoch in range(epochs):
    for step, (x_train, y_train) in enumerate(train_db):  #batch级别的循环 ，每个step循环一个batch
        with tf.GradientTape() as tape:  # with结构记录梯度信息
            y = tf.matmul(x_train, w1) + b1  # 神经网络乘加运算
            y = tf.nn.relu(y)    # relu激活函数
            y = tf.matmul(y,w2) + b2
            y_ = y_train  
            loss = tf.reduce_mean(tf.square(y_ - y))  # 采用均方误差损失函数mse = mean(sum(y-out)^2)
            loss_all += loss.numpy()  # 将每个step计算出的loss累加，为后续求loss平均值提供数据，这样计算的loss更准确
            loss_regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
            loss = loss + 0.03 * loss_regularization   #从结果上来看  不正则化貌似也不太影响
        # 计算loss对各个参数的梯度
        grads = tape.gradient(loss, [w1, b1, w2, b2])
        
        w1.assign_sub(lr * grads[0])  
        b1.assign_sub(lr * grads[1])  
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])

    # 每个epoch，打印loss信息
    print("Epoch {}, loss: {}".format(epoch, loss_all/6))  
    train_loss_results.append(loss_all / 6)  # 将4个step的loss求平均记录在此变量中
    loss_all = 0  # loss_all归零，为记录下一个epoch的loss做准备
    
a = tf.constant([[0.380472,-0.21714]],tf.float32)   # 结果导入模型
y1 = tf.matmul(a, w1) + b1
y1 = tf.nn.relu(y1)   
y1 = tf.matmul(y1,w2) + b2
print(y1)
print("*******predict*******")
# xx在-3到3之间以步长为0.01，yy在-3到3之间以步长0.01,生成间隔数值点
xx, yy = np.mgrid[-3:3:.1, -3:3:.1]
# 将xx, yy拉直，并合并配对为二维张量，生成二维坐标点
grid = np.c_[xx.ravel(), yy.ravel()]
grid = tf.cast(grid, tf.float32)
# 将网格坐标点喂入神经网络，进行预测，probs为输出
probs = []
for x_predict in grid:
    # 使用训练好的参数进行预测
    h1 = tf.matmul([x_predict], w1) + b1
    h1 = tf.nn.relu(h1)
    y = tf.matmul(h1, w2) + b2  # y为预测结果
    probs.append(y)
probs = np.array(probs).reshape(xx.shape)
plt.contour(xx, yy, probs, levels=[.5])
plt.show()
