---
title: 约束问题
categories:
  - Math Analysis
  - Optimization
tags:
  - Math
  - Analysis
  - Optimization
  - Constrained
  - Farkas
date: 2019-07-21 00:46:35
updated: 2019-07-21 00:46:35
toc: true
mathjax: true
comments: true
description: 约束问题
---

##	约束问题局部解

$$\begin{array}{l}
\min & f(x), x \in R^n \\
s.t. & c_i(x) = 0, i \in E = \{1,2,\cdots,l\}, \\
& c_i(x) \leq 0, i \in I = \{l,l+1,\cdots,l+m\}
\end{array}$$

-	对于一般约束优化问题，记其可行域为

	$$
	D = \{x| c_i(x) = 0, i \in E, c_i(x) \leq 0, i \in I\}
	$$

-	若 $\forall x^{*} \in D, \exists \epsilon$，使得当 $x \in D, \|x - x^{*}\| \leq \epsilon$ 时，总有

	$$ f(x) \geq f(x^{*}) $$

	则称$x^{*}$为约束问题的局部解，简称为最优解

-	若 $x \in D, 0 < \|x - x^{*}\| \leq \epsilon$ 时，总有

	$$ f(x) > f(x^{*})$$

	则称$x^{*}$是约束问题的严格局部最优解

##	约束问题局部解一阶必要条件

###	定理1

> - 设 $a_1,a_2,\cdots,a_m$ 和 $w \in R^n$，$C$ 定义如下 $$
		C = \{v |\sum_{i=1}^m \lambda_i a_i, \lambda_i \geq 0,
		i=1,2,\cdots,m \}
	$$
	若 $w \notin C$，则存在超平面 $d^T w = 0$，分离 $C$ 和 $w$，即 $$
	\begin{align*}
		d^T w & \leq 0 \\
		d^T w & > 0
	\end{align*}$$

-	显然C是闭凸集，则$\exists d \in R^n, d \neq 0$，
	$\alpha \in R$，使得

	$$\begin{array}{l}
	d^T x \leq \alpha, &  \forall x \in C \\
	d^T w > \alpha &
	\end{array}$$

-	又C是锥，有$0 \in C$，所以$\alpha \geq 0$，即$d^T w > 0$

-	若$\exists \bar x \in C, d^T \bar x > 0$，则
	$\forall \lambda \geq 0, \lambda \bar x \in C$，则有

	$$
	\lambda d^T \bar x \leq \alpha
	$$

	而$\lambda \rightarrow \infty$，左端趋于无穷，矛盾

###	Farkas引理

> - 设$a_1,a_2,\cdots,a_m$和$w \in R^n$，则以下两个系统有且
	仅有一个有解
> > -	系统I：存在$d$满足
		$$\begin{align*}
		a_i^T d & \leq 0, & i=1,2,\cdots,m \\
		w^T d & > 0
		\end{align*}$$
> > -	系统II：存在非负常数$\lambda_1,\cdots,\lambda_m$使得
		$$
		w =\sum_{i=1}^m \lambda_i a_i
		$$

-	若系统II有解，则系统I无解

	-	若系统II有解，即存在$\lambda_1,...,\lambda_m$且
		$\lambda_i \geq 0,i=1,2,\cdot,m$，使得

		$$
		w = \sum_{i=1}^m \lambda_i a_i
		$$

	-	若系统I有解，则有

		$$
		0 < w^T d = \sum_{i=1}^m \lambda_i a_i^T d \leq 0
		$$

		矛盾，因此系统I无解

-	若系统II无解，则系统I有解

	-	系统II误解，构造闭凸锥

		$$
		C = \{v |\sum_{i=1}^m \lambda_i a_i, \lambda_i \geq 0,
			i=1,2,\cdots,m \}
		$$

		显然$w \notin C$

	-	由**定理1**，存在d满足

		$$\begin{array}{l}
		d^T x \leq 0, & \forall x \in C, \\
		d^T w & > 0
		\end{array}$$

> - 此定理就是**点要么在凸锥C内、边缘（系统II），要么在凸锥
	外（系统I）**

####	推论1

> - 设$a_1,a_2,\cdots,a_m$和$w \in R^n$，则以下系统有且仅有
	一个有解
> > -	系统I：存在d满足
		$$\begin{array}{l}
		a_i^T d \leq 0, &  i=1,2,\cdots,m \\
		d_j \geq 0, &  j=1,2,\cdots,n \\
		w^T d > 0 &
		\end{array}$$
> > -	系统II：存在非负常数$\lambda_1,...,\lambda_m$使得
		$$
		w \leq \sum_{i=1}^m \lambda_i a_i
		$$

-	若系统II有解，则系统I无解
	-	若系统I有解，取d带入矛盾

-	若系统II无解，则系统I有解
	-	若系统I无解
#todo

####	推论2

> - 设$a_1,a_2,\cdots,a_{l+m}$和$w \in R^n$，则以下两个系统
	有且进一有一个存在解
> > -	存在d满足
		$$\begin{array}{l}
		a_i^T d = 0, & i=1,2,\cdots,l \\
		a_i^T d \leq 0, & i=l+1,l+2,\cdots,l+m \\
		w^T d > 0
		\end{array}$$
> > -	存在常数$\lambda_1,\lambda_2,\cdots,\lambda_{l+m}$
		且$\lambda_i \geq 0, i=l+1, l+2, \cdots, l+m$使得
		$$
		w = \sum_{i+1}^{l+m} \lambda_i a_i
		$$

##	迭代求解

###	参数部分更新

参数部分更新：每次更新一个或一组待估参数

-	应用场合
	-	适合待估参数较少、同时估计较慢，待估参数较多可能更新
		速度慢，往往需要多次迭代更新参数
	-	一般用在机器学习算法中比较多

-	特点（某些算法）

	-	良好的并行特性：能够同时更新多个参数
		-	*Alternating Direction Method of Multipliers*

	-	采用贪心策略的算法：可能无法得到最优解
		-	前向回归
		-	深度学习：网络层次太深，有些算法采用*固化*部分
			网络结构，估计剩余部分

	-	能够平衡全局、局部：得到较好的解
		-	LARS


