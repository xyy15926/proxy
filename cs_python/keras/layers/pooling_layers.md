---
title: 池化层
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
description: 池化层
---

##	MaxPooling1D

```python
keras.layers.pooling.MaxPooling1D(
	pool_size=2/int,
	strides=None/int,
	padding="valid"
	data_format=None
)
```

对时域1D信号进行最大值池化

-	参数

	-	`pool_size`：整数，池化窗口大小
	-	`strides`：整数或None，下采样因子，例如设2将会使得
		输出shape为输入的一半，若为None则默认值为pool_size。

##	MaxPooling2D

```python
keras.layers.pooling.MaxPooling2D(
	pool_size=(2, 2),
	strides=None/int/(int),
	padding="valid"/"same",
	data_format=None
)
```

为空域2D信号施加最大值池化 

##	MaxPooling3D层

```python
keras.layers.pooling.MaxPooling3D(
	pool_size=(2, 2, 2),
	strides=None/int/(int),
	padding="valid"/"same",
	data_format=None
)
```

为3D信号（空域或时空域）施加最大值池化

##	AveragePooling1D层

```python
keras.layers.pooling.AveragePooling1D(
	pool_size=2,
	strides=None,
	padding="valid"
	data_format=None
)
```
对1D信号（时域）进行平均值池化

##	AveragePooling2D层

```python
keras.layers.pooling.AveragePooling2D(
	pool_size=(2, 2),
	strides=None,
	padding="valid",
	data_format=None
)
```

为2D（空域）信号施加平均值池化


##	AveragePooling3D层

```python
keras.layers.pooling.AveragePooling3D(
	pool_size=(2, 2, 2),
	strides=None,
	padding="valid",
	data_format=None
)
```

为3D信号（空域或时空域）施加平均值池化

##	GlobalMaxPooling1D层

```python
keras.layers.pooling.GlobalMaxPooling1D(
	data_format="channels_last"
)
```

对于1D（时间）信号的全局最大池化

##	GlobalAveragePooling1D层

```python
keras.layers.pooling.GlobalAveragePooling1D(
	data_forma="channels_last"
)
```

为时域信号施加全局平均值池化

##	GlobalMaxPooling2D层

```python
keras.layers.pooling.GlobalMaxPooling2D(
	data_format=None
)
```
为空域信号施加全局最大值池化

##	GlobalAveragePooling2D层

```python
keras.layers.pooling.GlobalAveragePooling2D(
	data_format=None
)
```

为2D（空域）信号施加全局平均值池化

