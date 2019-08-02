---
title: 局部连接层LocallyConnceted
tags:
  - Python
  - Keras
categories:
  - Python
  - Keras
date: 2019-02-20 23:58:15
updated: 2019-02-17 11:57:07
toc: true
mathjax: true
comments: true
description: 局部连接层LocallyConnceted
---

LocallyConnnected和Conv差不多，只是Conv每层共享卷积核，
这里不同位置卷积核独立

##	LocallyConnected1D层

```python
keras.layers.local.LocallyConnected1D(
	filters,
	kernel_size,
	strides=1,
	padding="valid",
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

类似于`Conv1D`，单卷积核权重不共享

##	LocallyConnected2D层

```python
keras.layers.local.LocallyConnected2D(
	filters,
	kernel_size,
	strides=(1, 1),
	padding="valid",
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

类似`Conv2D`，区别是不进行权值共享

-	说明
	-	输出的行列数可能会因为填充方法而改变
-	例
	```python
	model = Sequential()
	model.add(LocallyConnected2D(64, (3, 3), input_shape=(32, 32, 3)))
		# apply a 3x3 unshared weights convolution with 64 output filters on a 32x32 image
		# with `data_format="channels_last"`:
		# now model.output_shape == (None, 30, 30, 64)
		# notice that this layer will consume (30*30)*(3*3*3*64) + (30*30)*64 parameters

	model.add(LocallyConnected2D(32, (3, 3)))
		# now model.output_shape == (None, 28, 28, 32)
		# add a 3x3 unshared weights convolution on top, with 32 output filters:
	```


