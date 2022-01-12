import tensorflow as tf
from sklearn import datasets
import numpy as np

x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())
])

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),  #优化器
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  #损失函数
              metrics=['sparse_categorical_accuracy'])   #准确率

model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)   # validation测试集验证

model.summary()
