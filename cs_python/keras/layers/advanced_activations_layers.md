---
title: 高级激活层
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
description: 高级激活层
---

### LeakyReLU

```python
keras.layers.LeakyReLU(alpha=0.3)
```

带泄漏的修正线性单元。

-	返回值：当神经元未激活时，它仍可以赋予其一个很小的梯度
	-	`x < 0`：`alpha * x`
	-	`x >= 0`：`x`

-	输入尺寸
	-	可以是任意的。如果将该层作为模型的第一层，需要指定
		`input_shape`参数（整数元组，不包含样本数量的维度）

-	输出尺寸：与输入相同

-	参数
	-	`alpha`：`float >= 0`，负斜率系数。

-	参考文献
	-	[Rectifier Nonlinearities Improve Neural Network Acoustic Models](https://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf)

### PReLU

```python
keras.layers.PReLU(
	alpha_initializer='zeros',
	alpha_regularizer=None,
	alpha_constraint=None,
	shared_axes=None
)
```

参数化的修正线性单元。

-	返回值
	-	`x < 0`：`alpha * x`
	-	`x >= 0`：`x`

-	参数
	-	`alpha_initializer`: 权重的初始化函数。
	-	`alpha_regularizer`: 权重的正则化方法。
	-	`alpha_constraint`: 权重的约束。
	-	`shared_axes`: 激活函数共享可学习参数的轴。
		如果输入特征图来自输出形状为
		`(batch, height, width, channels)`
		的2D卷积层，而且你希望跨空间共享参数，以便每个滤波
		器只有一组参数，可设置`shared_axes=[1, 2]`

-	参考文献
	-	[Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)

###	ELU

```python
keras.layers.ELU(alpha=1.0)
```

指数线性单元

-	返回值
	-	`x < 0`：`alpha * (exp(x) - 1.)`
	-	`x >= 0`：`x`

-	参数
	-	`alpha`：负因子的尺度。

-	参考文献
	-	[Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289v1)

### ThresholdedReLU

```python
keras.layers.ThresholdedReLU(theta=1.0)
```

带阈值的修正线性单元。

-	返回值
	-	`x > theta`：`x`
	-	`x <= theta`：0

-	参数
	-	`theta`：`float >= 0`激活的阈值位。

-	参考文献
	-	[Zero-Bias Autoencoders and the Benefits of Co-Adapting Features](http://arxiv.org/abs/1402.3337)

### Softmax

```python
keras.layers.Softmax(axis=-1)
```

Softmax激活函数

-	参数
	-	`axis`: 整数，应用 softmax 标准化的轴。

### ReLU

```python
keras.layers.ReLU(max_value=None)
```

ReLU激活函数

-	参数
	-	`max_value`：浮点数，最大的输出值。



