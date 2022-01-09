import tensorflow as tf
import numpy as np
import csv

x_data = []
y_data = []
with open("e:\\python homework\\tf_learn\\tensorflow_learning_notes\\regularization\\dot.csv",'r') as f:
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
x_data = tf.constant(x_data)
y_data = tf.constant(y_data)
x_data = tf.cast(x_data,tf.float32)
y_data = tf.cast(y_data,tf.float32)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2()),
    tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2()),
])

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

model.fit(x_data, y_data, batch_size=50, epochs=800)

model.summary()