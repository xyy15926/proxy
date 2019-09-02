---
title: Loss Function
tags:
  - Machine Learning
  - Loss Function
categories:
  - Machine Learning
date: 2019-08-01 01:48:22
updated: 2019-08-01 01:48:22
toc: true
mathjax: true
comments: true
description: Loss Function
---

##	损失函数设计

-	损失函数可以视为**模型与真实的距离**的度量
	-	因此损失函数设计关键即，寻找可以代表模型与真实的距离
		的统计量
	-	同时为求解方便，应该损失函数最好应满足导数存在

-	对有监督学习，**“真实”**已知，可以直接设计损失函数

-	对无监督学习，“真实”未知，需要给定**“真实标准”**
	-	NLP：需要给出语言模型
	-	EM算法：熵最大原理

##	常用损失函数

###	0-1 Loss

0-1损失函数

$$
L(y, f(x)) = \left \{ \begin{align*}
	1, & y \neq f(x) \\
	0, & y = f(x)
\end{align*} \right.
$$

-	适用场合
	-	二分类：Adaboost
	-	多分类：Adaboost.M1

### Hinge Loss

合页损失函数

$$\begin{align*}
L(y, f(x)) & = [1 - yf(x)]_{+} \\
[z]_{+} & = \left \{ \begin{array}{l}
	z, & z > 0 \\
	0, & z \leq 0
\end{array} \right.
\end{align*}$$

> - $y \in \{-1, +1\}$

-	合页损失函数是0-1损失函数的上界
-	合页损失函数要求分类不仅正确，还要求确信度足够高损失才
	为0，即对学习有更高的要求
-	适用场合
	-	二分类：线性支持向量机

###	Quadratic Loss

平方损失函数

$$
L(y, f(x)) = (y - f(x))^2
$$

-	适用场合
	-	回归预测：线性回归

###	Absolute Loss

绝对损失函数

$$
L(y, f(x)) = |y-f(x)|
$$

-	适用场合
	-	回归预测

###	Logarithmic Loss

对数损失函数（负对数极大似然损失函数）

$$
L(y, P(y|x)) = -logP(y|x)
$$

-	适用场合
	-	多分类：贝叶斯生成模型、逻辑回归

###	Exponential Loss

指数函数函数

$$
L(y, f(x)) = exp\{-yf(x)\}
$$

-	适用场合
	-	二分类：前向分步算法

###	Pseudo Loss

伪损失：考虑个体$(x_i, y_i)$

-	$h(x_i, y_i)=1, \sum h(x_i, y)=0$：完全正确预测
-	$h(x_i, y_i)=0, \sum h(x_i, y)=1$：完全错误预测
-	$h(x_i, y_i)=1/M$：随机预测（M为分类数目）

据此构造伪损失

$$
L(y, f(x)) = \frac 1 2 \sum_{y^{(j)} \ neq f(x)}
	w_j (1 - f(x, y) + f(x, y^{(j)}))
$$

> - $w_j$：样本个体错误标签权重，对不同个体分布可不同
> - $f(x, y^{(j)}$：分类器将输入$x$预测为第j类$y^{(j)}$的
	置信度

-	伪损失函数考虑了预测**标签**的权重分布
	-	通过改变此分布，能够更明确的关注难以预测的个体标签，
		而不仅仅个体

-	伪损失随着分类器预测准确率增加而减小
	-	分类器$f$对所有可能类别输出置信度相同时，伪损失最大
		达到0.5，此时就是随机预测
	-	伪损失大于0.5时，应该将使用$1-f$

-	适用场景
	-	多分类：Adaboost.M2

###	Cross Entropy

交叉熵损失

$$\begin{align*}
L(y, f(x)) & = -ylog(f(x)) \\
& = - \sum_{k=1}^K y_k log f(x)_k
\end{align*}$$

> - $y$：one-hot编码实际值
> - $f(x)$：各类别预测概率
> - $K$：分类数目

-	指数激活函数时：相较于二次损失，收敛速度更快

-	$y$为one-hot编码时，交叉熵损失可以视为对数损失
	（负对数极大似然函数）

	$$\begin{align*}
	L(y, f(x)) & = \sum_{x \in X} log \prod_{k=1}^K
		f(x)_k^{y_k} \\
	& = \sum_{x \in X} \sum_{k=1}^K f(x)_k^{y_k}
	\end{align*}$$

-	适合场合
	-	分类

> - 熵详细参见*machine_learning/reference/model_evaluation*

####	收敛速度对比

-	二次损失对$w$偏导

	$$
	\frac {\partial L} {\partial w} = (\sigma(z) - y)
		\sigma^{'}(z) x
	$$

	> - $\sigma$：sigmoid、softmax激活函数
	> - $z = wx + b$

	-	考虑到sigmoid函数输入值绝对值较大时，其导数较小
	-	激活函数输入$z=wx+b$较大时，$\sigma^{'}(z)$较小，
		更新速率较慢

-	Softmax激活函数时，交叉熵对$w$偏导

	$$\begin{align*}
	\frac {\partial L} {\partial w} & = -y\frac 1 {\sigma(z)}
		\sigma^{'}(z) x \\
	& = y(\sigma(z) - 1)x
	\end{align*}$$

-	特别的，对sigmoid二分类

	$$\begin{align*}
	\frac {\partial L} {\partial w_j} & = -(\frac y {\sigma(z)}
		- \frac {(1-y)} {1-\sigma(z)}) \sigma^{'}(z) x \\
	& = -\frac {\sigma^{'}(z) x} {\sigma(z)(1-\sigma(z))}
		(\sigma(z) - y) \\
	& = x(\sigma(z) - y)
	\end{align*}$$

	-	考虑$y \in \{(0,1), (1,0)\}$、$w$有两组
	-	带入一般形式多分类也可以得到二分类结果

