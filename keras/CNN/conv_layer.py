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


'''
在卷积神经网络中  通常还会有以下元素伴随者卷积层
1. 批标准化（BN)
对数据进行按batch进行批标准化处理，是数据回归到0为均值 1为标准差的分布，使得数据通过激活函数后更能明显体现数据微小变化带来的差异
提升区别能力（如sigmoid激活函数 0附近的数据更贴近线性变化）
2. 激活函数（Activation）
3. 池化（pooling）
用于减少特征数据量，降低数据冗余，常分为最大值池化和均值池化，分别作用为提取图片纹理和保留背景特征
4. 舍弃（dropout）
神经网络训练时，为缓解过拟合，可以使用按比例舍弃部分神经元，等再使用神经网络时再恢复

卷积->特征提取器（CBAPD)
'''