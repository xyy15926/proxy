---
title: Newton's Method
tags:
  - Optimization
  - Unconstrained
  - Newton's
categories:
  - Optimization
  - Unconstrianed Optimization
date: 2019-06-04 01:18:34
updated: 2019-06-04 01:18:34
toc: true
mathjax: true
comments: true
description: 牛顿法
---

##	Newton法

###	思想

-	若$x^{ * }$是无约束问题局部解，则有

	$$\nabla f(x^{ * }) = 0$$

	可求解此问题，得到无约束问题最优解

-	原始问题是非线性，考虑求解其线性逼近，在初始点$x^{(1)}$
	处泰勒展开

	$$
	\nabla f(x) \approx \nabla f(x^{(1)})
		+ \nabla^2 f(x^{(1)})(x - x^{(1)})
	$$

	解得

	$$
	x^{(2)} = x^{(1)} - (\nabla^2 f(x^{(1)}))^{-1}
		\nabla f(x^{(1)})
	$$

	作为$x^{ * }$的第二次近似

-	不断迭代，得到如下序列

	$$
	x^{(k+1)} = x^{(k)} + d^{(k)}
	$$

	> - $d^{(k)}$：Newton方向，是满足以下方程组解
		$$
		\nabla^2 f(x^{(k)}) d = -\nabla
			f(x^{(k)}
		$$

###	算法

1.	初始点$x^{(1)}$、精度要求$\epsilon$，置k=1

2.	若$\|\nabla f(x^{(k)})\| \leq \epsilon$，停止计算
	，得到最优解$x^{(k)}$，否则求解

	$$
	\nabla^2 f(x^{(k)}) d = -\nabla
		f(x^{(k)}
	$$

	得到$d^{(k)}$

3.	置

	$$x^{(k+1)} = x^{(k)} + d^{(k)}, k = k+1$$

	转2

###	特点

-	优点
	-	产生点列$\{x^{k}\}$若收敛，则具有二阶收敛速率
	-	具有二次终止性，事实上对正定二次函数，一步即可收敛

-	缺点
	-	可能会在某步迭代时目标函数值上升
	-	当初始点$x^{(1)}$距离最优解$x^{ * }$时，产生的点列
		可能不收敛，或者收敛到鞍点
	-	需要计算Hesse矩阵
		-	计算量大
		-	Hesse矩阵可能不可逆，算法终止
		-	Hesse矩阵不正定，Newdon方向可能不是下降方向

##	阻尼/修正Newton法

-	克服Newton法目标函数值上升的缺点
-	一定程度上克服点列可能不收敛缺点

###	算法

1.	初始点$x^{(1)}$、精度要求$\epsilon$，置k=1

2.	若$\|\nabla f(x^{(k)})\| \leq \epsilon$，停止计算
	，得到最优解$x^{(k)}$，否则求解

	$$
	\nabla^2 f(x^{(k)}) d = -\nabla
		f(x^{(k)}
	$$

	得到$d^{(k)}$

3.	一维搜索，求解一维问题

	$$
	\arg\min_{\alpha} \phi(\alpha) = f(x^{(k)} +
		\alpha d^{(k)})
	$$

	得到$\alpha_k$，置

	$$x^{(k+1)} = x^{(k)} + \alpha_k d^{(k)}, k = k+1$$

	转2

##	其他改进

-	针对Newton法、修正Newton法中Hesse矩阵可能不正定的改进

###	结合最速下降方向

将Newton方向和最速下降方向结合

-	设$\theta_k$是$d_N^{(k)}, -\nabla f(x^{(k)})$之间
	夹角，显然希望$\theta < \frac \pi 2$

-	则置限制条件$\eta$，取迭代方向

	$$d^{(k)} = \left \{ \begin{array}{l}
	d_N^{(k)}, & cos\theta_k \geq \eta \\
	-\nabla f(x^{(k)}), & 其他
	\end{array} \right.$$

###	*Negative Curvature*

当Hesse矩阵非正定时，选择负曲率下降方向$d^{(k)}$（一定存在）

-	Hesse矩阵非正定时，一定存在负特征值、相应特征向量$u$

-	可以取负曲率下降方向

	$$
	d^{(k)} = -sign(u^T \nabla f(x^{(k)})) u
	$$

> - $x^{(k)}$处负曲率方向$d^{(k)}$满足
	$$
	d^{(k)T} \nabla^2 f(x^{(k)}) d^{(k)} < 0
	$$

###	修正Hesse矩阵

取$d^{(k)}$为以下方程的解

$$
(\nabla^2 f(x^{(k)}) + v_k I) d =
	-\nabla f(x^{k})
$$

> - $v_k$：大于$\nabla^2 f(x^{(k)})$最大负特征值
	绝对值


