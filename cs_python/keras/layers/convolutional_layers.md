---
title: 卷积层
categories:
  - Python
  - Keras
tags:
  - Python
  - Keras
  - Machine Learning
  - Layer
  - Convolution
date: 2019-02-20 23:58:15
updated: 2019-02-17 11:57:07
toc: true
mathjax: true
comments: true
description: 卷积层
---

## Conv1D

```python
keras.layers.convolutional.Conv1D(
	filters(int),
	kernel_size(int),
	strides=1,
	padding='valid',
	dilation_rate=1,
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

一维卷积层（即时域卷积）

-	说明
	-	用以在一维输入信号上进行邻域滤波
	-	作为首层时，需要提供关键字参数`input_shape`
	-	该层生成将输入信号与卷积核按照单一的空域（或时域）
		方向进行卷积
	-	可以将Convolution1D看作Convolution2D的快捷版

-	参数

	-	`filters`：卷积核的数目（即输出的维度）

	-	`kernel_size`：整数或由单个整数构成的list/tuple，
		卷积核的空域或时域窗长度

	-	`strides`：整数或由单个整数构成的list/tuple，为卷积
		步长
		-	任何不为1的strides均与任何不为1的dilation_rate
			均不兼容

	-	`padding`：补0策略

	-	`activation`：激活函数

	-	`dilation_rate`：整数或由单个整数构成的list/tuple，
		指定dilated convolution中的膨胀比例
		-	任何不为1的dilation_rate均与任何不为1的strides
			均不兼容

	-	`use_bias`：布尔值，是否使用偏置项

	-	`kernel_initializer`：权值初始化方法
		-	预定义初始化方法名的字符串
		-	用于初始化权重的初始化器（参考initializers）

	-	`bias_initializer`：偏置初始化方法
		-	为预定义初始化方法名的字符串
		-	用于初始化偏置的初始化器

	-	`kernel_regularizer`：施加在权重上的正则项，为
		Regularizer对象

	-	`bias_regularizer`：施加在偏置向量上的正则项

	-	`activity_regularizer`：施加在输出上的正则项

	-	`kernel_constraints`：施加在权重上的约束项

	-	`bias_constraints`：施加在偏置上的约束项

-	输入：形如`(batch, steps, input_dim)`的3D张量

-	输出：形如`(batch, new_steps, filters)`的3D张量
	-	因为有向量填充的原因，`steps`的值会改变

##	Conv2D

```python
keras.layers.convolutional.Conv2D(
	filters,
	kernel_size,
	strides=(1, 1),
	padding='valid',
	data_format=None,
	dilation_rate=(1, 1),
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

二维卷积层，即对图像的空域卷积

-	说明
	-	该层对二维输入进行滑动窗卷积
	-	当使用该层作为第一层时，应提供

-	参数

	-	`filters`：卷积核的数目（即输出的维度）

	-	`kernel_size`：单个整数或由两个整数构成的list/tuple，
		卷积核的宽度和长度
		-	如为单个整数，则表示在各个空间维度的相同长度

	-	`strides`：单个整数或由两个整数构成的list/tuple，
		卷积的步长
		-	如为单个整数，则表示在各个空间维度的相同步长
		-	任何不为1的strides均与任何不为1的dilation_rate
			均不兼容

	-	`padding`：补0策略

	-	`activation`：激活函数

	-	`dilation_rate`：单个或两个整数构成的list/tuple，
		指定dilated convolution中的膨胀比例
		-	任何不为1的dilation_rate均与任何不为1的strides
			均不兼容

-	输入：`(batch, channels, rows, cols)`
	（"channels_first"）4D张量

-	输出：`(batch, filters, new_rows, new_cols)`
	（"channels_first"）4D张量

	-	输出的行列数可能会因为填充方法而改变

##	SeparableConv2D

```python
keras.layers.convolutional.SeparableConv2D(
	filters,
	kernel_size,
	strides=(1, 1),
	padding='valid',
	data_format=None,
	depth_multiplier=1,
	activation=None,
	use_bias=True,
	depthwise_initializer='glorot_uniform',
	pointwise_initializer='glorot_uniform',
	bias_initializer='zeros',
	depthwise_regularizer=None,
	pointwise_regularizer=None,
	bias_regularizer=None,
	activity_regularizer=None,
	depthwise_constraint=None,
	pointwise_constraint=None,
	bias_constraint=None
)
```

该层是在深度方向上的可分离卷积。

-	说明
	-	首先按深度方向进行卷积（对每个输入通道分别卷积）
	-	然后逐点卷积，将上步卷积结果混合到输出通道中
	-	直观来说，可分离卷积可以看做讲一个卷积核分解为两个小
		卷积核，或看作Inception模块的一种极端情况

-	参数

	-	`depth_multiplier`：按深度卷积的步骤中，每个输入通道
		使用（产生）多少个输出通道

	-	`depthwise_regularizer`：按深度卷积的权重上的正则项

	-	`pointwise_regularizer`：按点卷积的权重上的正则项

	-	`depthwise_constraint`：按深度卷积权重上的约束项

	-	`pointwise_constraint`：在按点卷积权重的约束项


-	输入：`(batch, channels, rows, cols)`4DT
	（"channels_first")

-	输出：`(batch, filters, new_rows, new_cols)`4DTK
	（"channels_first"）

	-	输出的行列数可能会因为填充方法而改变

##	Conv2DTranspose

```python
keras.layers.convolutional.Conv2DTranspose(
	filters,
	kernel_size,
	strides=(1, 1),
	padding="valid",
	output_padding=None/int/tuple,
	data_format=None,
	activation=None,
	use_bias=True,
	kernel_initializer="glorot_uniform",
	bias_initializer="zeros",
	kernel_regularizer=None,
	bias_regularizer=None,
	activity_regularizer=None,
	kernel_constraint=None,
	bias_constraint=None
)
```

该层是反卷积操作（转置卷积）

-	说明
	-	通常发生在用户想要对普通卷积的结果做反方向的变换
	-	参考文献
		-	[A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285)
		-	[Transposed convolution arithmetic](http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html#transposed-convolution-arithmetic)
		-	[Deconvolutional Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)

-	参数
	-	`output_padding`：指定输出的长、宽padding
		-	必须小于相应的`stride`

-	输入：`(batch, rows, cols, channels)`4DT
	（"channels_last")

-	输出：`(batch, new_rows, new_cols, filters)`4DT
	（"channels_last"）

	-	输出的行列数可能会因为填充方法而改变

##	Conv3D

```python
keras.layers.convolutional.Conv3D(
	filters,
	kernel_size,
	strides=(1, 1, 1),
	padding='valid',
	data_format=None,
	dilation_rate=(1, 1, 1),
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

三维卷积对三维的输入（视频）进行滑动窗卷积

-	输入：`(batch, channels, conv_dim1, conv_dim2, conv_dim3)`
	5D张量（"channnels_first"）

##	Cropping1D

```python
keras.layers.convolutional.Cropping1D(
	cropping=(1, 1)/tuple/int
)
```
在时间轴上对1D输入（即时间序列）进行裁剪

-	参数
	-	`cropping`：指定在序列的首尾要裁剪掉多少个元素
		-	单值表示首尾裁剪相同

-	输入：`(batch, axis_to_crop, features)`的3DT

-	输出：`(batch, cropped_axis, features)`的3DT

##	Cropping2D

```python
keras.layers.convolutional.Cropping2D(
	cropping=((0, 0), (0, 0)),
	data_format=None
)
```

对2D输入（图像）进行裁剪

-	说明
	-	将在空域维度，即宽和高的方向上裁剪

-	参数

	-	`cropping`：长为2的整数tuple，分别为宽和高方向上头部
		与尾部需要裁剪掉的元素数
		-	单值表示宽高、首尾相同
		-	单元组类似

-	输入：`(batch, rows, cols, channels)`4DT（"channels_last"）

-	输出：`(batch, cropped_rows, cropped_cols, channels)`

```python
	# Crop the input 2D images or feature maps
model = Sequential()
model.add(Cropping2D(cropping=((2, 2), (4, 4)),
                     input_shape=(28, 28, 3)))
	# now model.output_shape == (None, 24, 20, 3)
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Cropping2D(cropping=((2, 2), (2, 2))))
	# now model.output_shape == (None, 20, 16, 64)
```

##	Cropping3D

```python
keras.layers.convolutional.Cropping3D(
	cropping=((1, 1), (1, 1), (1, 1)),
	data_format=None
)
```

对3D输入（空间、时空）进行裁剪

-	参数

	-	`cropping`：长为3的整数tuple，分别为三个方向上头部
		与尾部需要裁剪掉的元素数

-	输入：`(batch, depth, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop)`
	（"channels_first"）

-	输出：`(batch, depth, first_cropped_axis, second_cropped_axis, third_cropped_axis)`

##	UpSampling1D

```python
keras.layers.convolutional.UpSampling1D(
	size=2/integer
)
```

在时间轴上，将每个时间步重复`size`次

-	参数
	-	`size`：轴上采样因子

-	输入：`(batch, steps, features)`的3D张量

-	输出：`(batch, upsampled_steps, feature)`的3D张量

##	UpSampling2D
```python
keras.layers.convolutional.UpSampling2D(
	size=(2, 2)/tuple/int,
	data_format=None
)
```

将数据的行和列分别重复`size[0]`和`size[1]`次

-	参数
	-	`size`：分别为行和列上采样因子

-	输入：`(batch, channels, rows, cols)`的4D张量
	（"channels_first"）

-	输出：`(batch, channels, upsampled_rows, upsampled_cols)`

##	UpSampling3D

```python
keras.layers.convolutional.UpSampling3D(
	size=(2, 2, 2)/tuple/int,
	data_format=None
)
```

将数据的三个维度上分别重复`size`次

-	说明
	-	本层目前只能在使用Theano为后端时可用

-	参数
	-	`size`：代表在三个维度上的上采样因子


-	输入：`(batch, dim1, dim2, dim3, channels)`5DT
	（"channels_last"）

-	输出：`(batch, upsampled_dim1, upsampled_dim2, upsampled_dim3, channels)`


##	ZeroPadding1D

```python
keras.layers.convolutional.ZeroPadding1D(
	padding=1/int
)
```

对1D输入的首尾端（如时域序列）填充0

-	说明
	-	以控制卷积以后向量的长度

-	参数
	-	`padding`：整数，在axis 1起始和结束处填充0数目

-	输入：`(batch, axis_to_pad, features)`3DT

-	输出：`(batch, paded_axis, features)`3DT

##	ZeroPadding2D

```python
keras.layers.convolutional.ZeroPadding2D(
	padding=(1, 1)/tuple/int,
	data_format=None
)
```

对2D输入（如图片）的边界填充0

-	说明
	-	以控制卷积以后特征图的大小

-	参数
	-	`padding`：在要填充的轴的起始和结束处填充0的数目

##	ZeroPadding3D
```python
keras.layers.convolutional.ZeroPadding3D(
	padding=(1, 1, 1),
	data_format=None
)
```
将数据的三个维度上填充0

-	说明
	-	本层目前只能在使用Theano为后端时可用

结束处填充0的数目

##	ZeroPadding3D
```python
keras.layers.convolutional.ZeroPadding3D(
	padding=(1, 1, 1),
	data_format=None
)
```
将数据的三个维度上填充0

-	说明
	-	本层目前只能在使用Theano为后端时可用

?时可用

