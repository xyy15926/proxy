---
title: 调参
categories:
  - Machine Learning
  - Parameter Adjustment
tags:
  - Machine Learning
  - Parameter Adjustment
  - Parameter Initialization
date: 2020-11-03 17:52:44
updated: 2020-11-03 17:52:44
toc: true
mathjax: true
description: 
---

##	参数初始化

-	合适参数初始化可以加速模型训练
	-	避免反向传播的梯度信息被放大、缩小
	-	导致出现梯度爆炸、消失

-	参数初始化值满足如下条件，则训练过程中能较好防止梯度信号
	被放缩

	$$\begin{align*}
	E(a^{(l-1)}) & = E(a^{(l)}) \\
	Var(a^{(l-1)}) & = Var(a^{(l)})
	\end{align*}$$

	-	激活值均值为0
	-	每层激活值方差保持一致

###	常数（零值）初始化

常数初始化：将所有权值初始化为常数

-	任意常数初始化方法性能都不好，甚至无法训练
	-	反向传播算法更新参数时，各参数各维度导数一致、更新
		后权值一致
	-	各神经元在演化过程中对称，无法学习不同特征，退化为
		单神经元

-	在激活函数选择线性激活函数时
	-	过大的初始化权重可能导致梯度爆炸
	-	国小的初始化值可能导致梯度消失

###	随机初始化

随机初始化：随机生成参数

-	权重$W$：均值0、方差1的正态分布生成，并乘以较小常数
	（如：0.01）
	-	权值被初始化不同值，解决零值初始化存在网络退化问题
	-	但较小权值可能导致梯度弥散，无法学习

-	偏置$b$：初始化为0
	-	帮助变换系统初始处于线性域，加快梯度传播

###	Xavier初始化

Xavier初始化：适合tanh激活函数的参数初始化方式

$$
Var(W^{(l)}) = \frac 2 {n^{(l-1)} + n^{(l)}}
$$

> - $n^{(l)}$：第$l$层神经元数量


###	He初始化

He初始化：适合ReLU激活函数的参数初始化方式

$$
Var(W^{(l)}) = \frac 1 {n^{(l)}}
$$


> - 基于Xavier初始化在ReLU上的改进，实际中二者都可以使用



##	超参搜索

###	*Random Search*

###	*Grid Search*

###	*Bayesian Optimization*



