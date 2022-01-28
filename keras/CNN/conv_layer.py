import tensorflow as tf

tf.keras.layers.Conv2D(
    filter = 2,  # 卷积核个数 也是该卷积层的深度
    kernel_size = 3,  # 卷积核尺寸
    strides = 1,  # 卷积核滑动步长
    padding = 'same',  # 是否全零填充 使输入输出图像尺寸一致  ”same“ 使用   ”valid“ 不适用（默认是不使用）
    activation = 'relu',  # 激活函数 与全连接层的设置一样 填写激活函数的名字
    input_shape = (5,5,1)  # 输入图像维度 （高，宽，通道数）,通常可省略
)

##  卷积层使用方式与此前一致  将定义好的卷积层加入Sequential
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filter = 2, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')
    # 配合其他内容
])