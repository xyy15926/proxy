---
title: Naive Bayes
tags:
  - Model
  - Linear Model
  - Bayes
  - Machine Learning
categories:
  - ML Model
  - Linear Model
date: 2019-07-13 23:24:10
updated: 2019-07-13 12:03:11
toc: true
mathjax: true
comments: true
description: Naive Bayes
---

##	*Naive Bayes Classifier* 

朴素贝叶斯：在训练数据集上学习联合概率分布$P(X,Y)$，利用后验
分布作为结果

-	朴素：条件概率分布有**条件独立性假设**，即特征在类别确定
	下条件独立

###	模型

-	输出**Y的先验概率分布**为

	$$
	P(Y = c_k), k = 1,2,\cdots,K
	$$

	> - 先验概率是指输出变量，即待预测变量的先验概率分布，
		反映其在无条件下的各取值可能性
	> - 同理所有的条件概率中也是以输出变量取值作为条件

-	条件概率分布为

	$$
	P(X=x|Y=c_k) = P(X^{(1)}=x^{(1)},\cdots,X^{(D)}=x^{(D)}|
		Y=c_k)
	$$

	> - $D$：用于分类特征数量

	其中有指数数量级的参数（每个参数的每个取值都需要参数）

-	因此对条件概率分布做**条件独立性假设**，即分类特征在类别
	确定条件下是独立的

	$$\begin{align*}
	P(X=x|Y=c_k) & = P(X^{(1)}=x^{(1)},\cdots,X^{(D)}=x^{(D)}
		|Y=c_k) \\
	& = \prod_{j=1}^D P(X^{(j)}=x^{(j)}|Y=c_k)
	\end{align*}$$

	-	条件独立性假设是比较强的假设，也是**朴素**的由来
	-	其使得朴素贝叶斯方法变得简单，但有时也会牺牲准确率

 -	以上即可得到联合概率分布$P(X,Y)$

	-	朴素贝叶斯学习到的联合概率分布$P(X,Y)$是数据生成的
		机制，即其为生成模型

###	策略

策略：选择使得后验概率最大化的类$c_k$作为最终分类结果

$$
P(Y=c_k|X=x) = \frac {P(Y=c_k, X=x)} {\sum_{i=1}^K
	P(Y=c_k, X=x)}
$$

> - $K$：输出类别数量

-	后验概率根计算根据贝叶斯定理计算

	$$\begin{align*}
	P(Y=c_k|X=x) & = \frac {P(X=x|Y=c_k)P(Y=c_k)}
		{\sum_{k=1}^K P(X=x|Y=c_k) P(Y=c_k)} \\
	& = \frac {P(Y=c_k) \prod_{j=1}^D P(X^{(j)}|Y=c_k)}
		{\sum_{k=1}^K P(Y=c_k) \prod_{j=1}^D P(X^{(j)}|Y=c_k)}
	\end{align*}$$

-	考虑上式中分母对所有$c_k$取值均相等，则最终分类器为

	$$
	y = \arg\max_{c_k} P(Y=c_k) \prod_{j=1}^D
		P(X^{(j)} = x^{(j)}|Y=c_k)
	$$

	-	即分类时，对给定输入$x$，将其归类为后验概率最大的类

####	策略性质

后验概率最大化等价于0-1损失的经验风险最小化

-	经验风险为

	$$
	\begin{align*}
	R_{emp}(f) & = E[L(Y, f(X))] \\
	& = E_x \sum_{k=1}^K L(y, c_k) P(c_k | X)
	\end{align*}
	$$

-	为使经验风险最小化，对训练集中每个$X=x$取极小化，对每个
	个体$(x,y)$有

	$$\begin{align*}
	f(x) & = \arg\min_{c_k} \sum_{k=1}^K L(y, c_k)
		P(c_k|X=x) \\
	& = \arg\min_{c_k} \sum_{k=1}^K P(y \neq c_k|X=x) \\
	& = \arg\min_{c_k} (1-P(y=c_k|X=x)) \\
	& = \arg\max_{c_k} P(y=c_k|X=x)
	\end{align*}$$
	
	即后验概率最大化

###	算法

####	极大似然估计

-	先验概率的极大似然估计为

	$$
	P(Y=c_k) = \frac {\sum_{i=1}^N I(y_i = c_k)} N,
		k=1,2,\cdots,K
	$$

-	条件概率的极大似然估计为

	$$
	P(X^{(j)}=a_{j,l}|Y=c_k) = \frac {\sum_{i=1}^N
		I(x_i^{(j)}=a_{j,l}, y_i=c_k)}
		{\sum_{i=1}^N I(y_i=c_k)} \\
		j=1,2,\cdots,N;l=1,2,\cdots,S_j;k=1,2,\cdots,K
	$$

	> - $a_{j,l}$；第j个特征的第l个可能取值
	> - $S_j$：第j个特征的可能取值数量
	> - $I$：特征函数，满足条件取1、否则取0

#####	算法

> - 输入：训练数据T
> - 输出：朴素贝叶斯分类器

1.	依据以上公式计算先验概率、条件概率

2.	将先验概率、条件概率带入，得到朴素贝叶斯分类器

	$$
	y = \arg\max_{c_k} P(Y=c_k) \prod_{j=1}^D
		P(X^{(j)} = x^{(j)}|Y=c_k)
	$$

####	贝叶斯估计

-	条件概率贝叶斯估计

	$$
	P(X^{(j)}=a_{j,l}|Y=c_k) = \frac {\sum_{i=1}^N
		I(x_i^{(j)}=a_{j,l}, y_i=c_k) + \lambda}
		{\sum_{i=1}^N I(y_i=c_k) + S_j \lambda} \\
		j=1,2,\cdots,N;l=1,2,\cdots,S_j;k=1,2,\cdots,K
	$$

	> - $\lambda \geq 0$

	-	$\lambda=0$时就是极大似然估计
	-	常取$\lambda=1$，此时称为*Laplace Smoothing*
	-	以上设计满足概率分布性质
		$$\begin{align*}
		P_{\lambda}(X^{(j)}=a_{j,l}|Y=c_k) \geq 0 \\
		\sum_{l=1}^{S_j} P_{\lambda}(X^{(j)}=a_{j,l}|Y=c_k)
			= 1
		\end{align*}
		$$

-	先验概率贝叶斯估计

	$$
	P_{\lambda}(Y=c_k) = \frac {\sum_{i=1}^N I(y_i = c_i)
		+ \lambda} {N + K\lambda}
	$$

> - 极大似然估计可能出现所需估计概率值为0，影响后验概率计算
	结果，贝叶斯估计能够避免这点

##	*Semi-Naive Bayes Classifier*

半朴素贝叶斯分类器：适当考虑部分特征之间的相互依赖信息

-	*Semi-Naive Bayes*可以视为是**利用规则对变量加权**，以
	此来体现相关变量的协同影响

	$$
	y = \arg\max_{c_k} P(Y=c_k) \prod_{j=1}^D
		\beta_j P(X^{(j)} = x^{(j)}|Y=c_k)
	$$

	-	特别的：权值为0/1即为变量筛选

###	*One-Depentdent Estimator*

独依赖估计：假设特征在类别之外最多依赖一个其他特征，这是半
朴素贝叶斯分类器中最常用的一种策略

$$
P(X=x|Y=c_k) = \prod_{j=1}^D P(X^{(j)}=x^{(j)} | Y=c_k, pa_j)
$$

> - $pa_j$：特征$X^{(j)}$依赖的父特征

-	若父特征已知，同样可以使用条件概率计算
	$P(X^{(j)}=x^{(j)} | Y=c_k, pa_j)$

	$$
	P(X^{(j)}=x^{(j)} | Y=c_k, pa_j) = \frac 
	{P(X^{(j)}=x^{(j)}, Y=c_k, pa_j)} {P(Y=c_k, pa_j)}
	$$

-	ODE形式半朴素贝叶斯分类器相应的策略为

	$$
	y = \arg\max_{c_k} P(Y=c_k) \prod_{j=1}^D
		P(X^{(j)} = x^{(j)}|Y=c_k, pa_j)
	$$

-	根据确定各特征父特征的不同做法，可以分为不同类型的独依赖
	分类器
	-	*Super-Parent ODE*：假设所有特征都依赖同一父特征
	-	*Averaged ODE*：类似随机森林方法，尝试将每个属性作为
		超父特征构建*SPODE*
	-	*Tree Augmented Naive Bayes*：基于最大带权生成树发展

####	SPODE

SPODE：每个特征只与其他唯一一个特征有依赖关系

$$
y = \arg\max_{c_k} P(Y=c_k, pa) \prod_{j=1}^D
	P(X^{(j)} = x^{(j)}|Y=c_k, pa)
$$

> - $pa$：所有特征共有的依赖父特征

####	AODE

AODE：以所有特征依次作为超父特征构建SPODE，以具有足够训练
数据支撑的SPODE集群起来作为最终结果

$$
y = \arg\max_{c_k} (\sum_{i=1}^D P(Y=c_k, X^{(i)})
	\prod_{j=1}^D P(X^{(j)} = x^{(j)}|Y=c_k, X^{(i)}))
$$

-	这里只选取训练数据足够，即取特征$X^{(i)}$某个取值的样本
	数量大于某阈值的SPODE加入结果

####	TAN

#####	TAN步骤

-	计算任意特征之间的互信息

	$$
	g(X^{(i)}, X^{(j)}| Y) = \sum P(X^{(i)}, X^{(j)} | Y=c_k)
		log \frac {P(X^{(i)}, X^{(j)} | Y=c_k)}
		{P(X^{(i)} | Y=c_k) P(X^{(j)} | Y=c_k)}
	$$

-	以特征为节点构建完全图，节点边权重设为相应互信息

-	构建此完全图的最大带权生成树
	-	挑选根变量
	-	将边设置为有向

-	加入预测节点$Y$，增加从$Y$到每个属性的有向边

#####	特点

-	条件互信息$g(X^{(i)}, X^{(j)}| Y)$刻画了特征在已知类别
	情况下的相关性

-	通过最大生成树算法，TAN仅保留了强相关属性之间的依赖性



