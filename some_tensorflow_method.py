import tensorflow as tf
a = tf.constant([[1,2,3],[2,3,4],[345,2,3],[1,2,4]],dtype=tf.float32)
print(a)
zero_tensor = tf.zeros([2,5])
print(zero_tensor)             # 生成全为0的张量， 参数为维度
fill_tensor = tf.fill([2,5],4)
print(fill_tensor)             # 生成特定维度张量 并用4填充

#--------
random_tensor = tf.random.normal([2,3], mean = 0.5, stddev = 1)  # 生成特定维度的随机数张量  其均值为0.5  标准差为1
print(random_tensor)
r_uniform_tensor = tf.random.uniform([2,3], minval = 0.2, maxval = 0.4)  # 生成特定范围内的随机数张量
print(r_uniform_tensor)

#---------
getmax = tf.reduce_max(a)  #取出a张量中的最大值  同理可以用min mean sum等获取最小 平均 加总
print('max: %s'%getmax)
getmax_a1 = tf.reduce_max(a,axis=1)   # axis表示对哪个维度进行操作  如为0时对第一维度即纵向 1时为横向   axis不定义时则对全部元素进行操作
print(getmax_a1)
fill_tensor = tf.cast(fill_tensor,tf.float32)
print('同纬度张量运算：%s'%tf.add(zero_tensor,fill_tensor))   # add substract multiply divide 加减乘除
print('张量平方: %s'%tf.square(fill_tensor))   # pow square sqrt  次方 平方 开方  
#tf.matmul(a,fill_tensor)   # 矩阵相乘


#------
sample = tf.constant([1.01, 3.1, -0.45])
sresult = tf.nn.softmax(sample)     #使n分类的n个输出通过softmax函数后得到符合概率分布的输出  从而可以根据
print('softmax后结果：%s'%sresult)