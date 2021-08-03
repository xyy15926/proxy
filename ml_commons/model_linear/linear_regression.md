---
title: 回归变量选择
categories:
  - ML Model
  - Linear Model
tags:
  - Machine Learning
  - ML Model
  - Linear Model
  - Regression
date: 2019-07-13 23:24:07
updated: 2021-07-16 14:50:43
toc: true
mathjax: true
comments: true
description: 回归变量选择
---

##	子集回归

> - 特征子集选择独立于回归模型拟合，属于封装器特征选择

###	最优子集

-	特点
	-	可以得到稀疏的模型
	-	但搜索空间离散，可变性大，稳定性差

###	*Forward Feature Elimination*

前向变量选择

####	步骤

-	初始变量集合$S_0 = \varnothing$
-	选择具有某种最优特性的变量进入变量集合，得到$S_1$
-	第j步时，从剩余变量中选择最优变量进入集合，得到$S_{j+1}$
-	若满足终止条件，则结束，否则重复上步添加变量
	-	j达到上限
	-	添加剩余变量均无法满足要求

###	*Backward Feature Elimination*

后向变量选择

####	步骤

-	初始变量集合$S_0$包含全部变量
-	从变量集合中剔除具有某种最差特性变量，得到$S_1$
-	第j步时，从剩余变量中剔除最差变量，得到$S_{j+1}$
-	若满足终止条件，则结束，否则重复上步添加变量
	-	j达到上限
	-	剔除剩余变量均无法满足要求

##	范数正则化约束

> - 回归过程中自动选择特征，属于集成特征选择

###	*Ridge Regression*

$$
\min_{\beta \in R^n} \left\{ ||y - X\beta||_2^2 +
	\lambda ||\beta||_2^2 \right\}
$$

-	在L2范数约束下最小化残差平方
-	作为连续收缩方法
	-	通过*bias-variance trade-off*，岭回归较普通最小二乘
		预测表现更好
	-	倾向于保留所有特征，无法产生疏系数模型

###	LASSO

$$
\min_{\beta \in R^n} \left\{ ||y - X\beta||_2^2 +
	\lambda||\beta||_1 \right\}
$$

能够选择部分特征，产生疏系数模型

-	p > n时，即使所有特征都有用，LASSO也只能从中挑选n个
-	如果存在相关性非常高的特征，LASSO倾向于只从该组中选择
	一个特征，而且是随便挑选的
	-	极端条件下，两个完全相同的特征函数，严格凸的罚函数
		（如Ridge）可以保证最优解在两个特征的系数相等，而
		LASSO的最优解甚至不唯一

###	Elastic Net

####	Naive Elastic Net

$$
\begin{align*}
& \min_{\beta \in R^n} \left\{ ||y - X\beta||_2^2 +
	\lambda_1||\beta||_1 + \lambda_2||\beta||_2^2 \right\} \\

\Rightarrow &
\min_{\beta^* \in R^p} \left\{ ||y - X^*\beta^*||_2^2 +
	\lambda^*||\beta^*||_1 \right\} \\

where: & y^* = \begin{pmatrix}
		y \\ \vec 0_p
	\end{pmatrix}	\\
& X^* = \frac 1 {\sqrt {1+\lambda^2}}
	\begin{pmatrix}
		X \\ \sqrt {\lambda_2} I_p
	\end{pmatrix} \\
& \beta^* = \sqrt {1+\lambda_2} \beta \\
& \lambda^* = \frac {\lambda_1} {1+\lambda_2} \\
\end{align*}
$$

-	弹性网在Lasso的基础上添加系数的二阶范数
	-	能同时做变量选择和连续收缩
	-	并且可以选择一组变量

-	传统的估计方法通过二阶段估计找到参数
	-	首先设置ridge系数$\lambda_2$求出待估参数$\beta$，
		然后做lasso的收缩
	-	这种方法有两次收缩，会导致估计偏差过大，估计不准

-	弹性网可以变换为LASSO，因而lasso的求解方法都可以用于
	elastic net

[elastic_net](http://www.stat.purdue.edu/~tlzhang/mathstat/ElasticNet.pdf)

##	*Least Angle Regression*

-	线性回归即找的一组系数能够用自变量的线性组合表示
	因变量

###	Forward Selection/Forward Stepwise Regression

-	从所有给定predictors中选择和y相关系数绝对值最大的变量
	$x_{j1}$，做线性回归

	-	对于标准化后的变量，相关系数即为变量之间的内积
	-	变量之间相关性越大，变量的之间的夹角越小，单个变量
		能解释得效果越好
	-	此时残差同解释变量正交

-	将上一步剩余的残差作为reponse，将剩余变量投影到残差上
	重复选择步骤

	-	k步之后即可选出一组变量，然后用于建立普通线性模型

-	前向选择算法非常贪心，可能会漏掉一些有效的解释变量，只是
	因为同之前选出向量相关

###	Forward Stagewise

前向选择的catious版本

-	和前向选择一样选择和y夹角最小的变量，但是每次只更新较小
	步长，每次更新完确认和y夹角最小的变量，使用新变量进行
	更新

	-	同一个变量可能会被多次更新，即系数会逐渐增加
	-	每次更新一小步，避免了前向选择的可能会忽略关键变量
