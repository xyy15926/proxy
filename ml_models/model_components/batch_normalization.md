---
title: Batch Normalization
tags:
  - 模型
  - 模型组件
categories:
  - 模型
  - 模型组件
date: 2019-07-29 21:16:01
updated: 2019-07-29 21:16:01
toc: true
mathjax: true
comments: true
description: Batch Normalization
---

##	Batch Normalization

*Batch Normalization*：规范化batch数据，使样本**各维度**
标准化，即均值为0、方差为1

$$\begin{align*}
\y & = BN_{\gamma, \beta}(z) = \gamma \odot \hat z + \beta \\
\hat z & = \frac {z - E(z)} {\sqrt {Var(z) + \epsilon}}
\end{align*}$$

> - $B$：mini-batch
> - $z, y$：**某层**输入向量、规范化后输入向量
	（即以个神经元中激活前标量值$z=Wx+b$为一维）
> - $\odot$：逐元素乘积
> - $E(x)$：均值使用移动平均均值
> - $Var(x)$：方差使用移动平均无偏估计
> - $\gamma, \beta$：待学习向量，用于**恢复网络的表示能力**
> - $\epsilon$：为数值计算稳定性添加

-	预测过程中各参数（包括均值、方差）为定值，BN仅仅对数据
	做了线性变换

-	规范化在每个神经元内部非线性激活前$z=Wu$进行，而不是
	[也]在上一层输出$u$上进行，即包含BN最终为

	$$
	z = act(BN(Wu))
	$$

	> - $act$：激活函数
	> - 偏置$b$可以被省略

	-	$u$的分布形状可以在训练过程中改变
	-	而$u$两次正则化无必要
	-	$z=Wu$分布更可能对称、稠密、类似高斯分布

-	对卷积操作，考虑卷积特性，不是只为激活函数（即卷积核）
	学习$\gamma, \beta$，而是为每个*feature map*学习
	（即每个卷积核、对每个特征图层分别学习）

> - BN可以视为*whitening*的简化，避免过高的运算代价、时间

###	背景

> - *internal covariate shift*：网络各层输入分布随前一层参数
	在训练过程中不断改变

-	网络中任意层都可以视为单独网络，则输入数据的分布性质对
	训练影响同样适用该层

-	考虑随机网络之前参数变化，输入数据分布可能对训练有负影响
	-	降低学习效率：要求更小的学习率
	-	参数初始化需要更复杂
	-	难以训练饱和非线性模型

###	用途

> - BN通过规范化输入数据各维度分布减少*ICS*

-	同时解耦梯度对参数值、初始值的依赖，利用梯度传播，
	允许使用更大的学习率提高学习效率

-	有正则化作用，提高模型泛化性能，减少对Dropout的需求

-	允许使用饱和非线性激活函数，而不至于停滞在饱和处

	$$\begin{align*}
	BN(Wu) & = BN((aW)u) \\
	\frac {\partial BN(aWu)} {\partial u} & = \frac
		{\partial BN(Wu)} {\partial u} \\
	\frac {BN(aWu)} {\partial aW} & = \frac 1 a \frac
		{\partial BN(Wu)} {\partial W}
	\end{align*}$$

	> - $a$：假设某层权重参数变动$a$倍

	-	激活函数函数输入不受权重$W$放缩影响
	-	梯度反向传播更稳定，权重$W$的Jacobian矩阵将包含接近
		1的奇异值，保持梯度稳定反向传播

-	方便模型迁移



