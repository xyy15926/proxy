---
title: Factorization Machines
tags:
  - Models
  - linear models
  - Machine Learning
  - Factorization Machine
categories:
  - ML Models
  - Linear Models
date: 2019-07-29 21:16:01
updated: 2019-07-29 21:16:01
toc: true
mathjax: true
comments: true
description: Factorization Machines
---

##	因子分解机

因子分解机：将变量交互影响因子化
（每个变量用隐向量代表、衡量其交叉影响）

$$
\hat y(x) := w_0 + \sum_{i=1}^m w_i x_i + \sum_{i=1}^m
	\sum_{j=i+1}^m <v_i, v_j> x_i x_j
$$

> - $w_0$：全局偏置
> - $w_i$：变量$i$权重
> - $w_{i,j} := <v_i, v_j>$：变量$i$、$j$之间交互项权重
> - $v_i$：$k$维向量，变量交叉影响因子

-	FM通过**因子化交互影响解耦交互项参数**

	-	即使没有足够数据也能较好估计高维稀疏特征交互影响参数
		-	无需大量有交互影响（交互特征取值同时非0）样本
		-	包含某交互影响数据也能帮助估计相关的交互影响
		-	**可以学习数据不存在的模式**

	-	可以视为embedding，特征之间关联性用embedding向量
		（隐向量）內积表示

-	参数数量、模型复杂度均为线性
	-	可以方便使用SGD等算法对各种损失函数进行优化
	-	无需像SVM需要支持向量，可以扩展到大量数据集

-	适合任何实值特征向量，对某些输入特征向量即类似
	*biased MF*、*SVD++*、*PITF*、*FPMC*

> - 另外还有d-way因子分解机，交互作用以PARAFAC模型因子化
	$$
	\hat y(x) := w_0 + \sum_{i=1}^n w_i x_i + \sum_{l=2}^d \sum_{i_1=1}
		\cdots \sum_{i_l=i_{l-1}+1}(\prod_{j=1}^l x_{i_j})
		(\sum_{f=1} \prod_{j=1}^l v_{i_j,f}^{(l)}) \\
	$$
> > -	$V^{(l)} \in R^{n * k_l}, k_l \in N_0^{+}$

###	模型表达能力

-	考虑任何正定矩阵$W$总可以被分解为$W=V V^T$，则$k$足够大
	时，FM总可以表达（还原）交叉项权重矩阵$W$

	-	FM是MF降维的推广，在用户-物品评分矩阵基础上集成其他
		特征
	-	特征组合发生所有变量之间

-	实际应该选取较小的$k$
	-	对较大$k$，稀疏特征没有足够数据估计复杂交叉项权重
		矩阵$W$
	-	限制FM的表达能力，模型有更好的泛化能力、交互权重矩阵

###	模型求解

$$\begin{align*}
\sum_{i=1}^m \sum_{j=i+1}^m <v_i, v_j> x_i x_j & = 
	\frac 1 2 \sum_{i=1}^m \sum_{j=i}^m <v_i, v_j> x_i x_j -
	\frac 1 2 \sum_{i=1}^m <v_i, v_i> x_i^2 \\
& = \frac 1 2 (x^T V^T V x - x^T diag(V^T V) x) \\
& = \frac 1 2 (\|Vx\|_2^2 - x^T diag(V^T V) x) \\
& = \frac 1 2 \sum_{f=1}^k ((\sum_{i=1}^m v_{i,f} x_i)^ 2
	- \sum_{i=1}^m v_{i,f}^2 x_i^2) \\
\end{align*}$$

> - $V = (v_1, v_2, \cdots, v_m)$
> - $x = (x_1, x_2, \cdots, x_m)^T$

-	模型计算复杂度为线性$\in O(kn)$

-	模型可以使用梯度下降类方法高效学习

	$$\begin{align*}
	\frac {\partial \hat y(x)} {\partial \theta} & = \left \{
		\begin{array}{l}
			1, & \theta := w_0 \\
			x_i, & \theta := w_i \\
			x_i Vx - v_i x_i^2& \theta := v_i
		\end{array} \right. \\
	& = \left \{ \begin{array}{l}
			1, & \theta := w_0 \\
			x_i, & \theta := w_i \\
			x_i \sum_{j=1}^m v_{j,f} x_j - v_{i,f} x_i^2,
				& \theta := v_{i,f}
		\end{array} \right.
	\end{align*}$$

> - 考虑到稀疏特征，內积只需计算非零值

###	模型适用

-	回归：直接用$\hat y(x)$作为回归预测值
-	二分类：结合logit损失、hinge损失优化
-	ranking：$\hat y(x)$作为得分排序，使用成对分类损失优化

##	Field-aware Factorization Machines

域感知因子分解机：在FM基础上考虑对特征分类，特征对其他类别
特征训练分别训练隐向量

$$\begin{align*}
\hat y(x) & = w_0 + \sum_{i=0}^m w_i x_i + \sum_{a=1}^m
	\sum_{b=a+1}^m <V_{a, f_b}, V_{b, f_a}> x_a x_b \\
& = w_0 + \sum_{i=1}^M \sum_{j=1}^{M_i} w_{i,j} x_{i,j} +
	\sum_{i=1}^M \sum_{j=1}^{M_i} \sum_{a=i}^M \sum_{b=1}^{M_i}
	<V_{i,j,a}, V_{a,b,i}> x_{i,j} x_{a,b}
\end{align*}$$

> - $m$：特征数量
> - $M, M_i$：特征域数量、各特征域中特征数量
> - $V_{i,j,a}$：特征域$i$中$j$特征对特征与$a$的隐向量
> - $V_{a, f_b}$：特征$x_a$对特征$b$所属域$f_b$的隐向量

-	FFM中特征都属于特定域，相同特征域中特征性质应该相同，
	一般的
	-	连续特征自己单独成域
	-	离散0/1特征按照性质划分，归于不同特征域

-	特征对其他域分别有隐向量表示**和其他域的隐含关系**
	-	考虑交互作用时，对不同域使用不同隐向量计算交互作用
	-	FFM中隐变量维度也远远小于FM中隐向量维度

###	算法

![ffm_steps](imgs/ffm_steps.png)

###	模型特点

-	模型总体类似FM，仅通过多样化隐向量实现细化因子分解
-	模型总体较FM复杂度大、参数数量多
	-	无法抽取公因子化简为线性
	-	数据量较小时可能无法有效训练隐向量


