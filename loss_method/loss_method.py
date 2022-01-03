'''
损失函数：旨在将预测值与实际值比较，并逐渐将预测值向实际值收敛，从而更新权重w和加权b

常用损失函数有
1. 均方误差
2. 交叉熵
3. 自定函数
'''
# y_实际值  y预测值
#均方误差    
#  loss = tf.reduce_mean(tf.square(y_ - y))

#交叉熵  表征两个概率分布之间的距离  H(y_,y) = -∑y_*lny
#  loss = tf.losses.categorical_crossentropy(y_,y)
#  loss = tf.nn.softmax_cross_entropy_with_logits(y_,y)    由于通常预测结果y的数据不符合概率分布  需先经过softmax函数才能进行交叉熵计算  此函数则可直接一步到位

#自定函数
#例如：由于使用均方误差默认了预测值与实际值比较过大与过小效果一样，在实际问题中有时候可能不是这样，因此需要自定函数来解决
