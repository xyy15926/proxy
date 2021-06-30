---
title: 凸优化问题
tags:
  - 最优化
categories:
  - 最优化
date: 2019-07-21 00:46:35
updated: 2019-07-21 00:46:35
toc: true
mathjax: true
comments: true
description: 凸优化问题
---

##	*Linear Programming*

###	数学模型

####	一般数学模型

线性规划问题（LP）可以抽象为一般的数学模型

$$\begin{array}{l}
(\min_x, \max_x) & S=c_1 x_1 + c_2 x_2 + \cdots + c_n x_n \\
s.t. & \left \{ \begin{array} {l}
		a_{1,1} x_1 + a_{1,2} x_2 + \cdots + a_{1,n} x_n (\geq = \leq) b_1 \\
		a_{2,1} x_1 + a_{2,2} x_2 + \cdots + a_{2,n} x_n (\geq = \leq) b_2 \\
		\vdots \\
		a_{m,1} x_1 + a_{m,2} x_2 + \cdots + a_{m,n} x_n (\geq = \leq) b_m
	\end{array} \right.
\end{array}$$

> - $S = c_1 x_1 + c_2 x_2 + \cdots + c_n x_n$：目标函数
> - $x_1, x_2, ..., x_n$：待求解变量
> - $b_i、c_i、a_{ij}$：实常数
> - $(\geq = \leq)$：在三种符号中取一种

####	标准形式

$$\begin{array}{l}
\min_x & S=c_1 x_1 + c_2 x_2 + \cdots + c_n x_n \\
s.t. & a_{1,1} x_1 + a_{1,2} x_2 + \cdots + a_{1,n} x_n = b_1 \\
& \vdots \\
& a_{t,1} x_1 + a_{t,2} x_2 + \cdots + a_{t,n} x_n = b_t \\
& a_{t+1,1} x_1 + a_{t+1,2} x_2 + \cdots + a_{t+1, n} x_n
	\leq b_{t+1} \\
& \vdots \\
& a_{t+l,1} x_1 + a_{t+l,2} x_2 + \cdots + a_{t+l, n} x_n
	\leq b_{t+l} \\
\end{array}$$

> - $\max_x$：目标函数取翻转换为$\min_x$
> - $\geq$：不等式左右取反转换为$\leq$

> - 线性规划一般模式都可以等价转换为标准形式

###	Simplex Method

单纯型法：利用线性规划极值点必然在单纯型顶点取得，不断迭代顶点求出极值

####	算法

-	初始化：标准化线性规划问题，建立初始表格
	-	最小化目标函数：目标函数系数取反，求极大
	-	不等式约束：加入松弛变量（代表不等式两端差值）
	-	变量非负：定义为两个非负变量之差
-	最优测试
	-	若目标行系数都为非负，得到最优解，迭代停止
	-	基变量解在右端列中，非基变量解为 0
-	确定主元列
	-	从目标行的前 $n$ 个单元格中选择一个非负单元格，确定主元列
	-	选择首个非负：解稳定，若存在最优解总是能取到
	-	选择绝对值最大：目标函数下降快，但有可能陷入死循环，无法得到最优解（不满足最优条件）
-	确定主元（分离变量）（行）
	-	对主元列所有正系数，计算右端项和其比值 $\Theta$ 比率
	-	最小 $\Theta$ 比率确定主元（行）（类似的为避免死循环，总是选择首个最小者）

-	转轴变换（建立新单纯形表）
	-	主元变 1：主元行所有变量除以主元
	-	主元列变 0：其余行减去其主元列倍主元行
	-	交换基变量：主元行变量标记为主元列对应变量

###	特点

-	算法时间效率
	-	极点规模随着问题规模指数增长，所以最差效率是指数级
	-	实际应用表明，对 $m$ 个约束、$n$ 个变量的问题，算法迭代次数在 $m$ 到 $3m$ 之间，每次迭代次数正比于 $nm$
-	迭代改进

###	Two-Phase Simplex Method

两阶段单纯形法：单纯型表中没有单元矩阵，无法方便找到基本可行解时使用

-	在给定问题的约束等式中加入人工变量，使得新问题具有明显可行解
-	利用单纯形法求解最小化新的线性规划问题

###	其他一些算法

-	大 M 算法
-	*Ellipsoid Method* 椭球算法
	-	算法时间效率
		-	可以在多项式时间内对任意线性规划问题求解
		-	实际应用效果较单纯形法差，但是最差效率更好
-	*Karmarkar* 算法
	-	内点法（迭代改进）

##	凸优化

$$\begin{array}{l}
\min_x & f(x) \\
s.t. & g_i(x) \leq 0, i=1，2,\cdots,k \\
& h_i(x) = 0, i=1,2,\cdots,l
\end{array}$$

> - $f(x), g(x)$：$R^n$ 上连续可微的凸函数
> - $h_i(x)$：$R^n$ 上仿射函数
> - 仿射函数：满足 $f(x)=ax+b, a \in R^n, b \in R, x \in R^n$

###	二次规划

$$\begin{array}{l}
\min_x & f(x)=\frac 1 2 x^TGx + c^Tx \\
s.t. & Ax \leq b
\end{array}$$

> - $G \in R^{n * n}$：$n$ 阶实对称矩阵
> - $A \in R^{m * n}$：$m * n$ 实矩阵
> - $b \in R^m$
> - $c \in R^n$

-	$G$ 正定
	-	此时目标函数 $f(x)$ 为凸函数
	-	凸二次规划
	-	问题有唯一全局最小值
	-	问题可可由椭球法在多项式时间内求解

-	$G$ 半正定
	-	此时目标函数 $f(x)$ 为凸函数
	-	半定规划
	-	若约束条件可行域不空，且目标函数在此可行域有下界，则问题有全局最小值

-	$G$非正定
	-	目标函数有多个平稳点（局部极小），*NP-hard* 问题

####	求解

-	椭球法
-	内点法
-	增广拉格朗日法
-	投影梯度法

###	二阶锥规划

$$\begin{array}{l}
\min_x & f^Tx \\
s.t. & \|A_ix + b_i \|_2 \leq c_i^Tx + d_i, i=1,2,\cdots,m  \\
	& Bx=g
\end{array}$$

> - $f \in R^n$
> - $A_i \in R^{n_i * n}$，$b_i \in R^{n_i}$，$c_i \in R^{n_i}$，$d_i \in R$
> - $B \in R^{p * n}$，$g \in R^n$

-	$A_i=0,i=1,\cdots,m$：退化为线性规划

-	一般的二阶规划可以转换为二阶锥规划

	$$
	X^TAX + qTX + C \leq 0 \Rightarrow
		\|A^{1/2}x + \frac 1 2 A^{-1/2}q\|^{1/2} \leq
		-\frac 1 4 q^TA^{-1}q - c
	$$

> - 二阶锥规划可以使用内点法很快求解（多项式时间）

##	非线性最小二乘

$$\begin{align*}
f(x) & = \frac 1 2 \sum_{i=1}^m r^2_i(x) \\
& = \frac 1 2 r(x) r^T(x)
\end{align*}$$

> - $r_i(x)$：通常为非线性函数
> - $r(x) = (r_1(x), \cdots, r_n(x))^T$
> - $x \in R^n, m \geq n$

-	考虑目标函数梯度、*Hesse* 矩阵

	$$\begin{align*}
	\nabla f(x) & = \sum_{i=1}^m \nabla r_i(x)
		r_i(x) \\
	& = J(x)^T r(x) \\

	\nabla^2 f(x) & = \sum_{i=1}^m \nabla r_i(x)
		r_i(x) + \sum_{i=1}^m r_i \nabla^2 r_i(x) \\
	& = J(x)^T J(x) + \sum_{i=1}^m r_i(x) \nabla^2 r_i(x)
	\end{align*}$$

> - $$
	J(x) = \begin{bmatrix}
	\frac {\partial r_1} {\partial x_1} &
		\frac {\partial r_1} {\partial x_2} & \cdots &
		\frac {\partial r_1} {\partial x_n} \\
	\frac {\partial r_2} {\partial x_1} &
		\frac {\partial r_2} {\partial x_2} & \cdots &
		\frac {\partial r_2} {\partial x_n} \\
	\vdots & \vdots & \ddots & \vdots \\
	\frac {\partial r_m} {\partial x_1} &
		\frac {\partial r_m} {\partial x_2} & \cdots &
		\frac {\partial r_m} {\partial x_n}
	\end{bmatrix}
	= \begin{bmatrix}
	\nabla r_1(x)^T \\
	\nabla r_2(x)^T \\
	\vdots \\
	\nabla r_m(x)^T \\
	\end{bmatrix}
	$$
	为$r(x)$的 *Jacobi* 矩阵

###	*Gauss-Newton* 法

-	为简化计算，略去 *Newton* 法中 *Hesse* 矩阵中 $\sum_{i=1}^m r_i(x) \nabla^2 r_i(x)$ 项，即直接求解方程组

	$$
	J(x^{(k)})^T J(x^{(k)}) d = -J(x^{(k)})^T r(x^{(k)})
	$$

-	求解同一般 *Newton* 法

####	特点

-	实际问题中
	-	局部解 $x^{ * }$ 对应的目标函数值 $f(x^{ * })$ 接近 0 时，采用 *Gauss-Newton* 法效果较好，此时
		-	$\|r(x^{(k)})\|$ 较小
		-	曲线$r_i(x)$接近直线
		-	$\nabla^2 r_i(x) \approx 0$
	-	否则效果一般

-	矩阵 $J(x^{(k)})^T J(x^{(k)})$ 是半正定矩阵
	-	当 *Jacobi* 矩阵列满秩时为正定矩阵，此时虽然 $d^{(k)}$ 是下降方向，但仍需类似修正牛顿法增加一维搜索策略保证目标函数值不上升

###	*Levenberg-Marquardt* 方法

-	考虑到 $J(x^{(k)})$ 中各列线性相关、接近线性相关，求解 *Newton-Gauss *方法中的方程组会出现困难，可以改为求解

	$$
	(J(x^{(k)})^T J(x^{(k)}) + vI) d = -J(x^{(k)})^T r(x^{(k)})
	$$

> - $v$：迭代过程中需要调整的参数，*LM* 方法的关键即如何调整

####	定理1

> - 若 $d(v)$ 是以上方程组的解，则 $\|d(v)\|^2$ 是 $v$ 的连续下降函数，且 $v \rightarrow +\infty, \|d(v)\| \rightarrow 0$

-	$J(x^{(k)})^T J(x^{(k)})$ 是对称半正定矩阵，则存在正交阵

	$$
	(P^{(k)})^T J(x^{(k)})^T J(x^{(k)}) P^{(k)} = \Lambda^{(k)}
	$$

-	则可以解出 $\|d(v)\|^2$

> - 增大 $v$ 可以限制 $\|d^{(k)}\|$，所以 *LM* 方法也被称为阻尼最小二乘法

####	定理2

> - 若 $d(v)$ 是以上方程的解，则 $d(v)$ 是 $f(x)$ 在 $x^{(k)}$ 处的下降方向，且 $v \rightarrow + \infty$ 时，$d(v)$ 的方向与 $-J(x^{(k)})^T r(x^{(k)})$ 方向一致

-	下降方向：$\nabla f(x^{(k)}) d(v) < 0$ 即可
-	方向一致：夹角余弦

> - $v$充分大时，*LM* 方法产生的搜索方向 $d^{(k)}$ 和负梯度方向一致

####	参数调整方法

> - 使用梯度、近似 *Hesse* 矩阵定义二次函数
	$$
	q(d) = f(x^{(k)}) + (J(x^{(k)})^T r(x^{(k)}))^T d + \frac 1 2 d^T (J(x^{(k)})^T J(x^{(k)})) d
	$$

-	其增量为

	$$\begin{align*}
	\Delta q^{(k)} & = q(d^{(k)}) - q(0) \\
	& = (J(x^{(k)})^T r(x^{(k)}))^T d^{(k)} + \frac 1 2
		(d^{(k)})^T (J(x^{(k)})^T J(x^{(k)})) d^{(k)}
	\end{align*}$$

-	目标函数增量

	$$\begin{align*}
	\Delta f^{(k)} & = f(x^{(k)} + d^{(k)}) - f(x^{(k)}) \\
	& = f(x^{(k+1)}) - f(x^{(k)})
	\end{align*}$$

-	定义$\gamma_k = \frac {\Delta f^{(k)}} {\Delta q^{(k)}}$

	-	$\gamma_k$接近1说明$\Delta f^{(k)}$接近$\Delta q^{(k)}$
		-	即$f(x^{(k)} + d^{(k+1)})$接近$q(d^{(k)})$
		-	即$f(x)$在$x^{(k)}$附近接近二次函数
		-	即使用Gauss-Newton方法求解最小二乘问题效果较好
		-	即LM方法求解时$v$参数应该较小

	-	$\gamma_k$接近0说明$\Delta f^{(k)}$与$\Delta q^{(k)}$
		近似程度不好
		-	$d^{(k)}$不应取得过大，应减少$d^{(k)}$得模长
		-	应该增加参数$v$进行限制
		-	迭代方向趋近于负梯度方向

	-	$\gamma_k$适中时，认为参数$v$选取合适，不做调整
		-	临界值通常为0.25、0.75

####	算法

1.	初始点 $x^{(1)}$、初始参数 $v$（小值）、精度要求 $\epsilon$，置 $k=k+1$

2.	若 $\|J(x^{(k)})^T r(x^{(k)})\| < \epsilon$，则停止计算，得到问题解 $x^{(k)}$，否则求解线性方程组

	$$
	(J(x^{(k)})^T J(x^{(k)}) + v_kI) d = -J(x^{(k)})^T
		r(x^{(k)})
	$$

	得到 $d^{(k)}$

3.	置 $x^{(k+1)} = x^{(k)} + d^{(k)}$，计算 $\gamma_k$

4.	考虑 $\gamma$
	-	$\gamma < 0.25$，置$v_{k+1} = 4 v_k$
	-	$\gamma > 0.75$，置$v_{k+1} = v_k / 2$
	-	否则置$v_{k+1} = v_k$

5.	置k=k+1，转2

##	其他问题

###	整形规划

整形规划：求线性函数的最值，函数包含若干**整数变量**，并且满足线性等式、不等式的有限约束

###	*Unregularized Least Squares Learning Problem*

$$
w_T = \frac \gamma n \sum_{i=0}^{T-1} (I - \frac \gamma n
	{\hat X}^T \hat X)^i {\hat X}^T \hat Y
$$

> - $\gamma$：被引入保证 $\|I - \frac \gamma n {\hat X}^T \hat X\| < 1$

####	策略

-	考虑

	$$
	\min_w I_s(w) = \frac 1 {2n} \|\hat X w - \hat Y\|^2
	$$

-	将$w_{t+1}$带入$I_s(w)$即可证明每次迭代$I_s(w)$减小

	$$
	w_0 = 0 \\
	w_{t+1} = (I - \frac \gamma n {\hat X}^T \hat X)w_t + \frac \gamma n {\hat X}^T \hat Y
	$$



