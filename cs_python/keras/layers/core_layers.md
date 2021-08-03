---
title: 常用层
categories:
  - Python
  - Keras
tags:
  - Python
  - Keras
  - Machine Learning
  - Layer
  - Dense
  - Flatten
  - Dropout
  -	Reshape
  - Activation
date: 2019-02-20 23:58:15
updated: 2019-02-17 11:57:07
toc: true
mathjax: true
comments: true
description: 常用层
---

常用层对应于core模块，core内部定义了一系列常用的网络层，包括
全连接、激活层等

##	Dense层

```python
keras.layers.core.Dense(
	units,
	activation=None,
	use_bias=True,
	kernel_initializer='glorot_uniform',
	bias_initializer='zeros',
	kernel_regularizer=None,
	bias_regularizer=None,
	activity_regularizer=None,
	kernel_constraint=None,
	bias_constraint=None
)
```

Dense就是常用的全连接层

-	用途：实现运算$output = activation(dot(input, kernel)+bias)$
	-	`activation`：是逐元素计算的激活函数
	-	`kernel`：是本层的权值矩阵
	-	`bias`：为偏置向量，只有当`use_bias=True`才会添加

-	参数

	-	`units`：大于0的整数，代表该层的输出维度。

	-	`activation`：激活函数

		-	为预定义的激活函数名（参考激活函数）
		-	逐元素（element-wise）的Theano函数
		-	不指定该参数，将不会使用任何激活函数
			（即使用线性激活函数：a(x)=x）

	-	`use_bias`: 布尔值，是否使用偏置项

	-	`kernel_initializer`：权值初始化方法
		-	预定义初始化方法名的字符串
		-	用于初始化权重的初始化器（参考initializers）

	-	`bias_initializer`：偏置向量初始化方法
	
		-	为预定义初始化方法名的字符串
		-	用于初始化偏置向量的初始化器（参考initializers）

	-	`kernel_regularizer`：施加在权重上的正则项，为
		Regularizer对象

	-	`bias_regularizer`：施加在偏置向量上的正则项，为
		Regularizer对象

	-	`activity_regularizer`：施加在输出上的正则项，为
		Regularizer对象

	-	`kernel_constraints`：施加在权重上的约束项，为	
		Constraints对象

	-	`bias_constraints`：施加在偏置上的约束项，为
		Constraints对象

-	输入

	-	形如`(batch_size, ..., input_dim)`的NDT，最常见情况
		为`(batch_size, input_dim)`的2DT
	-	数据的维度大于2，则会先被压为与`kernel`相匹配的大小

-	输出

	-	形如`(batch_size, ..., units)`的NDT，最常见的情况为
		$(batch_size, units)$的2DT

##	Activation层

```python
keras.layers.core.Activation(
	activation,
	input_shape
)
```

激活层对一个层的输出施加激活函数

-	参数

	-	`activation`：将要使用的激活函数
		-	预定义激活函数名
		-	Tensorflow/Theano的函数（参考激活函数）

-	输入：任意，使用激活层作为第一层时，要指定`input_shape`

-	输出：与输入shape相同

##	Dropout层

```python
keras.layers.core.Dropout(
	rate,
	noise_shape=None,
	seed=None
)
```

为输入数据施加Dropout

-	说明
	-	Dropout将在训练过程中每次更新参数时按一定概率`rate`
		随机断开输入神经元

	-	可以用于防止过拟合

	-	参考文献：[Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)


-	参数

	-	`rate`：0~1的浮点数，控制需要断开的神经元的比例

	-	`noise_shape`：整数张量，为将要应用在输入上的二值
		Dropout mask的shape

	-	`seed`：整数，使用的随机数种子

-	输入
	-	例：`(batch_size, timesteps, features)`，希望在各个
		时间步上Dropout mask都相同，则可传入
		`noise_shape=(batch_size, 1, features)`

##	Flatten层

```python
keras.layers.core.Flatten()
```

Flatten层用来将输入“压平”，把多维的输入一维化

-	常用在从卷积层到全连接层的过渡
-	Flatten不影响batch的大小。

```python
model = Sequential()
model.add(Convolution2D(64, 3, 3,
            border_mode='same',
            input_shape=(3, 32, 32)))
	# now: model.output_shape == (None, 64, 32, 32)

model.add(Flatten())
	# now: model.output_shape == (None, 65536)
```

##	Reshape层

```python
keras.layers.core.Reshape(
	target_shape,
	input_shape
)
```

Reshape层用来将输入shape转换为特定的shape

-	参数

	-	`target_shape`：目标shape，为整数的tuple，不包含样本
		数目的维度（batch大小）
		-	包含`-1`表示推断该维度大小

-	输入：输入的shape必须固定（和`target_shape`积相同）

-	输出：`(batch_size, *target_shape)`

-	例

	```python
	model = Sequential()
	model.add(Reshape((3, 4), input_shape=(12,)))
		# now: model.output_shape == (None, 3, 4)
		# note: `None` is the batch dimension

	model.add(Reshape((6, 2)))
		# now: model.output_shape == (None, 6, 2)

		# also supports shape inference using `-1` as dimension
	model.add(Reshape((-1, 2, 2)))
		# now: model.output_shape == (None, 3, 2, 2)
	```

##	Permute层

```python
keras.layers.core.Permute(
	dims(tuple)
)
```

Permute层将输入的维度按照给定模式进行重排

-	说明
	-	当需要将RNN和CNN网络连接时，可能会用到该层。

-	参数
	-	`dims`：指定重排的模式，不包含样本数的维度（即下标
		从1开始）

-	输出shape
	-	与输入相同，但是其维度按照指定的模式重新排列

-	例
	```python
	model = Sequential()
	model.add(Permute((2, 1), input_shape=(10, 64)))
		# now: model.output_shape == (None, 64, 10)
	```

##	RepeatVector层

```python
keras.layers.core.RepeatVector(
	n(int)
)
```

RepeatVector层将输入重复n次

-	参数
	-	`n`：整数，重复的次数

-	输入：形如`(batch_size, features)`的张量

-	输出：形如`(bathc_size, n, features)`的张量

-	例
	```python
	model = Sequential()
	model.add(Dense(32, input_dim=32))
		# now: model.output_shape == (None, 32)

	model.add(RepeatVector(3))
		# now: model.output_shape == (None, 3, 32)
	```

##	Lambda层

```python
keras.layers.core.Lambda(
	function,
	output_shape=None,
	mask=None,
	arguments=None
)
```

对上一层的输出施以任何Theano/TensorFlow表达式

-	参数

	-	`function`：要实现的函数，该函数仅接受一个变量，即
		上一层的输出

	-	`output_shape`：函数应该返回的值的shape，可以是一个
		tuple，也可以是一个根据输入shape计算输出shape的函数

	-	`mask`: 掩膜

	-	`arguments`：可选，字典，用来记录向函数中传递的其他
		关键字参数

-	输出：`output_shape`参数指定的输出shape，使用TF时可自动
	推断


-	例
	```python
	model.add(Lambda(lambda x: x ** 2))
		# add a x -> x^2 layer
	```

	```python
	# add a layer that returns the concatenation
	# of the positive part of the input and
	# the opposite of the negative part

	def antirectifier(x):
		x -= K.mean(x, axis=1, keepdims=True)
		x = K.l2_normalize(x, axis=1)
		pos = K.relu(x)
		neg = K.relu(-x)
		return K.concatenate([pos, neg], axis=1)

	def antirectifier_output_shape(input_shape):
		shape = list(input_shape)
		assert len(shape) == 2  # only valid for 2D tensors
		shape[-1] *= 2
		return tuple(shape)

	model.add(Lambda(antirectifier,
			 output_shape=antirectifier_output_shape))
	```

##	ActivityRegularizer层

```python
keras.layers.core.ActivityRegularization(
	l1=0.0,
	l2=0.0
)
```

经过本层的数据不会有任何变化，但会基于其激活值更新损失函数值

-	参数
	-	`l1`：1范数正则因子（正浮点数）
	-	`l2`：2范数正则因子（正浮点数）


##	Masking层

```python
keras.layers.core.Masking(mask_value=0.0)
```

使用给定的值对输入的序列信号进行“屏蔽”

-	说明
	-	用以定位需要跳过的时间步
	-	对于输入张量的时间步，如果输入张量在该时间步上都等于
		`mask_value`，则该时间步将在模型接下来的所有层
		（只要支持masking）被跳过（屏蔽）。
	-	如果模型接下来的一些层不支持masking，却接受到masking
		过的数据，则抛出异常

-	输入：形如`(samples,timesteps,features)`的张量


-	例：缺少时间步为3和5的信号，希望将其掩盖
	-	方法：赋值`x[:,3,:] = 0., x[:,5,:] = 0.`
	-	在LSTM层之前插入`mask_value=0.`的Masking层
		```python
		model = Sequential()
		model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
		model.add(LSTM(32))
		```

.`的Masking层
		```python
		model = Sequential()
		model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
		model.add(LSTM(32))
		```

```


