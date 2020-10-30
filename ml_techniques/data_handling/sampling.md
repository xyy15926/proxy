---
title: 抽样方法
tags:
  - Machine Learning
  - Data Preprocessing
  - Sampling
categories:
  - ML Technique
  - Data Preprocessing
date: 2020-08-17 21:20:05
updated: 2020-08-17 21:20:05
toc: true
mathjax: true
comments: true
description: 数据抽样技术
---

##	样本评价

样本质量：抽样样本与整体的相似性

$$\begin{align*}
J(S, D) & = \frac {1} {D} \sum_{k=1}^{r} J_{k}(S, D) \\
J_{k}(S, D) & = \sum_{j=1}^{N_k}(P_{Sj} - P_{Dj})
	log \frac {P_{Sj}} {P_{Dj}} \\
Q(s) & = exp(-J)
\end{align*}$$

> - $D$：数据集，包含$r$个属性
> - $S$：抽样样本集
> - $J_k=J(S, D)$：*Kullblack-Laible*信息量，数据集$S$、$D$
	在属性$k$上偏差程度，越小偏差越小
> - $Q(S) \in [0, 1]$：抽样集$S$在数据集$D$中的质量，越大
	样本集质量越高

###	说明

-	若整体$D$分布稀疏，容易得到$S$在某些数据点观测值数为0，
	得到$I(S, D) \rightarrow infty$

	-	可以把该点和附近的点频率进行合并，同时调整总体频率
		分布
	-	过度合并会导致无法有效衡量数据集局部差异性

-	对于连续型变量

	-	可以把变量进行适当分组：粗糙，不利于刻画数据集直接的
		局部差异
	-	计算数据集各个取值点的非参估计，如核估计、最近邻估计
		等，再在公式中用各自的非参估计代替相应频率，计算样本
		质量

-	数据包含多个指标时
	-	可以用多个指标的平均样本质量衡量整体样本质量
	-	也可以根据指标重要程度，设置不同的权重

##	测试集、训练集划分

-	测试集、训练集划分逻辑前提
	-	在样本量足够的情况下，减少部分样本量不会影响模型精度
	-	模型评价需要使用未参与建模数据验证，否则可能夸大模型
		效果

-	测试集、训练集划分作用
	-	测试集直接参与建模，其包含信息体现在模型中
	-	训练集仅仅用于评价模型效果，其包含信息**未被利用**，
	-	因此，若无评价、对比模型需求，或有其他无需划分测试集
		即可评价模型，则划分测试集无意义

###	*Hold Out*

旁置法：将样本集随机划分为训练集、测试集，只利用训练集训练
模型

-	适合样本量较大的场合
	-	减少部分训练数据对模型精度影响小
	-	否则大量样本未参与建模，影响模型精度
-	常用划分比例
	-	8:2
	-	7:3
-	旁置法建立模型可直接作为最终输出模型
	-	旁置法一般只建立一个模型
	-	且使用旁置法场合，模型应该和全量数据训练模型效果差别
		不大

###	*N-fold Cross Validation*

N折交叉验证：将数据分成N份，每次将其中一份作为测试样本集，
其余N-1份作为训练样本集

-	N折交叉验证可以视为旁置法、留一法的折中
	-	克服了旁置法中测试样本选取随机性的问题：每个样本都
		能作为测试样本
	-	解决了留一法计算成本高的问题：重复次数少

-	典型的“袋外验证”
	-	袋内数据（训练样本）、袋外数据（测试样本）分开

-	N折交叉验证会训练、得到N个模型，不能直接输出
	-	最终应该输出全量数据训练的模型
	-	N折建立N次模型仅是为了合理的评价模型效果，以N个模型
		的评价指标（均值）作为全量模型的评价

###	*Leave-One-Out Cross Validation*

留一法：每次选择一个样本作为测试样本集，剩余n-1个观测值作为
训练样本集，重复n次计算模型误差

-	可以看作是N折交叉验证的特例

##	样本重抽样

###	Bootstrap

重抽样自举：有放回的重复抽样，以模拟多组独立样本

-	对样本量为$n$的样本集$S$
-	做$k$次有放回的重复抽样
	-	每轮次抽取$n$个样本
	-	抽取得到样本仍然放回样本集中
-	得到$k$个样本容量仍然为$n$的随机样本$S_i，(i=1,2,...,k)$

###	过采样

> - *over-sampling*：过采样，小类数据样本增加样本数量

-	*synthetic minority over-sampling technique*：过采样
	算法，构造不同于已有样本小类样本
	-	基于距离度量选择小类别下相似样本
	-	选择其中一个样本、随机选择一定数据量邻居样本
	-	对选择样本某属性增加噪声，构造新数据

###	欠采样

> - *under-sampling*：欠采样，大类数据样本减少样本数量


