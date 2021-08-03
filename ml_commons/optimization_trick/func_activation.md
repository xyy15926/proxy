---
title: 激活函数
categories:
  - ML Optimization
  - Neural Network
tags:
  - Machine Learning
  - ML Model
  - Neural Network
  - Activation
  - Function
date: 2019-07-29 21:16:01
updated: 2021-07-16 15:05:31
toc: true
mathjax: true
comments: true
description: 激活函数
---

##	指数类

###	Sigmoid

将实数映射到(0, 1)区间

$$
sigmoid(z) = \frac 1 {1+e^{-z}}
$$

> - $z= wx+b$

-	用途
	-	隐层神经元输出
	-	二分类输出

-	缺点
	-	激活函数计算量大，BP算法求误差梯度时，求导涉及除法
	-	误差反向传播时容易出现梯度消失
	-	函数收敛缓慢

###	Hard_Sigmoid

计算速度比sigmoid激活函数快

$$
hard_signmoid(z) = \left \{ \begin {array} {l}
	0 & z < -2.5 \\
	1 & z > 2.5 \\
	0.2*z + 0.5 & -2.5 \leq z \leq 2.5 \\
\end {array} \right.
$$

> - $z= wx+b$

###	Softmax

主要用于多分类神经网络输出

$$
softmax(z_i) = \frac {e^{z_i}} {\sum_{k=1}^K e^{z_k}}
$$

> - $z_i = w_i x + b_i$：$(w_i, b_i)$组数同分类数量，和输入
	$x$维度无关

> - $K$：分类数目

-	工程意义：指数底
	-	可导$max$：拉开数值之间差距
	-	特征对输出结果为乘性：即$z_i$中输入增加会导致输出
		随对应权重倍数增加
	-	联合交叉熵损失避免导数溢出，提高数值稳定性

-	理论意义：概率论、最优化
	-	softmax符合最大熵原理
	-	假设各标签取值符合多元伯努利分布，而softmax是其
		link functiond的反函数#todo
	-	光滑间隔最大函数

> - Softmax回归参数$(w_i, b_i$$冗余，可以消去一组

###	Softplus

$$
softplus(z) = log(exp(z)+1)
$$

> - $z = wx + b$

###	Tanh

双曲正切函数

$$
\begin{align*}
tanh(z) & = \frac {sinhz} {coshz} \\
	& = \frac {e^z - e^{-z}} {e^z + e^{-z}} \\
\end{align*}
$$

> - $z = wx + b$
> - $\frac{\partial tanh(z)}{\partial z} = (1 - tanh(z))^2$
	：非常类似普通正切函数，可以简化梯度计算

##	线性类

###	Softsign

$$
softsign(z) = \frac z {abs(z) + 1)}
$$

###	ReLU

*Rectfied Linear Units*：修正线性单元

$$
relu(z, max) = \left \{ \begin{array} {l}
	0 & z \leq 0 \\
	z & 0 < x < max \\
	max & z \geq max \\
\end {array} \right.
$$

###	LeakyReLU

*Leaky ReLU*：带泄露的修正线性

$$
relu(z, \alpha, max) = \left \{ \begin {array} {l}
	\alpha z & z \leq 0 \\
	z & 0 < z < max \\
	max & z \geq max \\
\end {array} \right.
$$

> - $\alpha$：超参，建议取0.01

-	解决了$z < 0$时进入死区问题，同时保留了ReLU的非线性特性

###	Parametric ReLU

*PReLU*：参数化的修正线性

$$
prelu(z) = \left \{ \begin{array} {l}
	\alpha z & z < 0 \\
	z & z> 0 \\
\end{array} \right.
$$

> - $\alpha$：自学习参数（向量），初始值常设置为0.25，通过
	momentum方法更新

###	ThreshholdReLU

带阈值的修正线性

$$
threshhold_relu(z, theta)= \left \{ \begin{array} {l}
	z & z > theta \\
	0 & otherwise \\
\end{array} \right.
$$

###	Linear

线性激活函数：不做任何改变

##	线性指数类

###	Exponential Linear Unit

*Elu*：线性指数

$$
elu(z, \alpha) =
\left \{ \begin{array} {l}
	z & z > 0 \\
	\alpha (e^z - 1) & x \leqslant 0 \\
\end{array} \right.
$$

> - $\alpha$：超参

-	$x \leq 0$时，$f(x)$随$x$变小而饱和
	-	ELU对输入中存在的特性进行了表示，对缺失特性未作定量
		表示

> - 网络深度超超过5层时，ELU相较ReLU、LReLU学习速度更快、
	泛化能力更好

###	Gausssion Error Liear Unit

GELU：ReLU的可导版本

###	Selu

可伸缩指数线性激活：可以两个连续层之间保留输入均值、方差

-	正确初始化权重：`lecun_normal`初始化
-	输入数量足够大：`AlphaDropout`
-	选择合适的$\alpha, scale$值

$$
selu(z) = scale * elu(z, \alpha)
$$


##	梯度消失

激活函数导数太小（$<1$），压缩**误差（梯度）**变化


