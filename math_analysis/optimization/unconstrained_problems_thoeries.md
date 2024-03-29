---
title: 无约束优化
categories:
  - Math Analysis
  - Optimization
tags:
  - Math
  - Analysis
  - Optimization
  - Unconstrained
date: 2019-07-21 00:46:35
updated: 2019-07-21 00:46:35
toc: true
mathjax: true
comments: true
description: 无约束优化
---

##	无约束局部解

$$
minf(x), x \in R^n
$$

> - 若存在$x^{ * } \in R^n, \epsilon > 0, \forall x \in R^n$
	使得$\|x - x^{ * }\| < \epsilon$时，恒有
> - $f(x) \geq f(x^{ * })$：则称$x^{ * }$为f(x)的
	*local minimum point/solution*（局部极小点/局部解）
> - $f(x) > f(x^{ * })$：则称$x^{ * }$为f(x)的
	严格局部极小点/局部解

##	最优性条件

###	*First-Order Necessary Condtion*

> - 无约束问题局部解的一阶必要条件：设f(x)有连续的一阶偏导，
	弱$x^{ * }$是无约束问题的局部解，则
	$$ \triangledown f(x{* }) = 0$$

###	*Second-Order Necessary Condition*

> - 无约束问题局部解的二阶必要条件：设f(x)有连续二阶偏导，
	若$x^{ * }$是无约束问题的局部解，则
> > -	$$\triangledown f(x^{ * }) = 0$$
> > -	$$\triangledown^2 f(x^{ * })半正定$$

###	*Second-Order Sufficient Condition*

> - 无约束问题局部解的二阶充分条件：设f(x)有连续二阶偏导，
	若在$x^{ * }$处满足以下，则x^{ * }是无约束问题的
	**严格局部解**
> > -	$$\triangledown f(x^{ * }) = 0$$
> > -	$$\triangledown^2 f(x^{ * })正定$$

##	下降算法

迭代算法：将当前迭代点**向正确方向**移动**一定步长**，然后
检验目标值是否满足一定要求

-	**方向**、**步长**就是不同优化算法主要关心的两个方面
-	还关心算法的*rate of convergence*（收敛速率）

###	一般下降算法框架

1.	取初始点$x^{(1)}$，置精度要求$\epsilon$，置k=1

2.	若在点$x^{(k)}$处满足某个终止准则，则停止计算，得无约束
	优化问题最优解$x^{(k)}$，否则**适当地选择**$x^{(k)}$处
	**搜索方向**

3.	进行**适当的一维搜索**，求解一维问题
	$$
	\arg\min_{\alpha} \phi(\alpha) =
		f(x^{(k)} + \alpha d^{(k)})
	$$

4.	置k=k+1，转2

要使下降算法可行，需要确定

-	某点出搜索方向
	-	负梯度方向
	-	Newton方向：求方向的时候已确定步长，也可用做步长搜索
	-	拟Newton方向
-	求步长地一维搜索方式
	-	试探法
		-	0.618法
		-	Fibonacci方法（分数法）
		-	二分法
	-	插值法
		-	三点二次插值法
		-	二点二次插值法
		-	两点三次插值法
	-	非精确一维搜索方法
		-	Glodstein方法
		-	Armijo方法
		-	Wolfe-Powell方法
-	算法终止准则
	-	$\|\triangledown f(x^{(k)})\| < \epsilon$
	-	$\|x^{(k+1)} - x^{(k)}\| < \epsilon$
	-	$|f(x^{(k+1)}) - f(x^{(k)})| < \epsilon$

	> - 实际计算中最优解可能永远无法迭代达到，应该采用较弱
		终止准则

###	算法收敛性

> - 收敛：序列$\{x^{(k)}\}$或其一个子列（仍记$\{x^{(k)}\}$）
	满足
	$$
	\lim_{k \rightarrow \infty} x^{(k)} = x^{ * }
	$$
> > -	$x^{ * }$：无约束问题局部解

但是这样强的结果难以证明

-	往往只能证明$\{x^{(k)}\}$的任一聚点的稳定点
-	或是更弱的
	$$
	\lim_{k \rightarrow \infty} inf
		\|\triangledown f(x^{(k)}) \| = 0
	$$

> - 局部收敛算法：只有初始点充分靠近极小点时，才能保证产生
	序列收敛
> - 全局收敛算法：对任意初始点，产生序列均能收敛

####	收敛速率

设序列$\{x^{(k)}\}$收敛到$x^{ * }$，若以下极限存在

$$
\lim _ {k \rightarrow \infty} \frac {\|x^{(k+1)} - x^{*}\|}
	{\|x^{(k)} - x^{*}\|} = \beta
$$

> - $0 < \beta < 1$：线性收敛
> - $\beta = 0$：超线性收敛
> - $\beta = 1$：次线性收敛（收敛速率太慢，一般不考虑）

####	算法的二次终止性

> - 二次终止性：若某算法对任意正定二次函数，从任意初始点出发
	，都能经过有限步迭代达到其极小点，则称该算法有二次终止性

具有二次终止性的算法被认为时好算法，否则计算效果较差，原因

-	正定二次目标函数有某些好的性质，好的算法应该能在有限步内
	达到其极小点

-	对于一个一般的目标函数，若其在极小点处的Hesse矩阵
	$\triangledown f(x^{( * )})$，则由泰勒展开式得到
	$$\begin{align*}
	f(x) & = f(x^{*}) + \triangledown f(x^ {*})^T(x - x^{*}) \\
		& + \frac 1 2 (x - x^{*})^T \triangledown^2 f(x^{*})
			(x - x^{*}) \\
		& + o(\|x - x^{*}\|^2)
	\end{align*}$$
	即目标函数f(x)在极小点附近与一个正定二次函数近似，所以对
	正定二次函数好的算法，对一般目标函数也应该具有较好的性质

