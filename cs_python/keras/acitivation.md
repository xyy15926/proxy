---
title: 激活函数
tags:
  - Python
  - Keras
categories:
  - Python
  - Keras
date: 2019-03-21 17:27:37
updated: 2019-02-17 11:57:07
toc: true
mathjax: true
comments: true
description: 激活函数
---

-	通过设置单独的激活层实现
	```python
	from keras.layers import Activation, Dense

	model.add(Dense(64))
	model.add(Activation('tanh'))
	```

-	也可以在构造层对象时通过传递`activation`参数实现

	```python
	model.add(Dense(64, activation='tanh'))
	```

-	可以通过传递一个逐元素运算的Theano/TensorFlow/CNTK函数
	来作为激活函数

	```python
	from keras import backend as K

	model.add(Dense(64, activation=K.tanh))
	model.add(Activation(K.tanh))
	```
##	softmax


```python
softmax(
	x(tensor),
	axis=-1/int
)
```

Softmax激活函数

-	参数

	-	`x`：张量
	-	`axis`：整数，代表softmax所作用的维度

-	返回值：softmax变换后的张量。

-	异常：

	-	`ValueError`： In case `dim(x) == 1`

##	elu

```python
elu(
	x,
	alpha=1.0/num
)
```

指数线性单元

-	参数
	-	`alpha`：一个标量，表示负数部分的斜率。

-	返回值：线性指数激活
	-	如果 `x > 0`，返回值为 `x`
	-	如果 `x < 0` 返回值为 `alpha * (exp(x)-1)`

-	参考文献
	-	[Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)

##	selu

```python
selu(x)
```

可伸缩的指数线性单元（SELU）。


-	参数
	-	`x`: 一个用来用于计算激活函数的张量或变量。

-	返回值：可伸缩线性指数激活
	-	可伸缩的指数线性激活：`scale * elu(x, alpha)`
		（`scale, alpha`应该是预定义常数？）

-	注意
	-	与「lecun_normal」初始化方法一起使用。
	-	与 dropout 的变种「AlphaDropout」一起使用。
	-	只要正确初始化权重（参见 `lecun_normal` 初始化方法）
		并且输入的数量「足够大」（参见参考文献获得更多信息）
		，选择合适的`alpha`和`scale`的值，就可以在两个连续层
		之间保留输入的均值和方差

-	参考文献
	-	[Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)

##	softplus

```python
softplus(x)
```

Softplus 激活函数

-	返回值：$log(exp(x) + 1)$

##	softsign


```python
softsign(x)
```

Softsign 激活函数

-	返回值：$x / (abs(x) + 1)$

##	relu


```python
relu(
	x,
	alpha=0.0,
	max_value=None
)
```

线性修正单元。

-	参数
	-	`alpha`：负数部分的斜率。默认为 0。
	-	 `max_value`：输出的最大值

-	返回值：线性修正单元激活
	-	`x > 0`：返回值为 `x`
	-	`x < 0`：返回值为 `alpha * x`
	-	如果定义了`max_value`，则结果将截断为此值

##	tanh

```python
tanh(x)
```

双曲正切激活函数

##	sigmoid


```python
sigmoid(x)
```

Sigmoid 激活函数

##	hard_sigmoid

```python
hard_sigmoid(x)
```

Hard sigmoid 激活函数。

-	说明
	-	计算速度比 sigmoid 激活函数更快。

-	返回值：Hard sigmoid 激活：
	-	`x < -2.5`：返回 0
	-	`x > 2.5`：返回 1
	-	`-2.5 <= x <= 2.5`：返回 `0.2 * x + 0.5`

##	linear


```python
linear(x)
```

线性激活函数（即不做任何改变）


