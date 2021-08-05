---
title: Projected Gradient Descent
categories:
  - Math Analysis
  - Optimization
tags:
  - Math
  - Analysis
  - Optimization
  - Constrained
  - Project
date: 2019-07-21 00:46:35
updated: 2019-07-21 00:46:35
toc: true
mathjax: true
comments: true
description: Projected Gradient Descent
---

##	Projected Gradient Descent

###	受限优化问题

$$
\min_{x \in C} f(x)
$$

> - $C \subseteq R^d$：受限凸集

投影梯度下降：采用后处理的方式，将迭代位置拉回到约束条件内

-	使用一般下降算法进行位置更新，新位置$x_{t+1}^{'}$可能
	不再满足约束条件

-	为使新位置$x_{t+1}^{'}$符合受限集合，可以选择在$L_2$范数
	下距离受限集合$C$最近的的点
	$x_{t+1}=\arg\min_{x \in C} \|x - x_{t+1}^{'}\|$作为下步
	真正迭代位置

##	线性约束

###	*Projection Matrix*

> - 投影矩阵：矩阵$P \in R^{n*n}$，若满足$P^T = P, P^2 = P$
> - 若$A \in R^{m*n}$为行满秩矩阵，则$A$的零空间为
	$L_A = {x \in R^{n} | Ax = 0}$，对应正交空间为
	$L_A^{\perp} = {A^T y | y \in R^m}$

对$\forall x \in R^n$进行正交分解

$$\begin{align*}
\forall x \in R^n, x & = x_1 + x_2, x_1 \in L_A,
	x_2 \in L_A^{\perp} \\
x_1 & = P_A x
\end{align*}$$

> - $P_A = I - A^T (A A^T)^{-1} A$：$A$的投影矩阵
> - 投影矩阵$P_A$可由点对线性约束的投影定义，利用拉格朗日
	求解

证明

$$\begin{align*}
x_1 & = x - x_2 = x - A^T y \\
A x_1 & = A x - A A^T y \\
\Rightarrow y & = (A A^T)^{-1} A (x - x_1) \\
\Rightarrow x_1 & = x - A^T[(A A^T)^{-1} A (x - x_1)] \\
& = x - A^T (A A^T)^{-1} A x - A^T (A A^T)^{-1} A x_1 \\
& = (I - A^T (A A^T)^{-1} A) x = P_A x
\end{align*}$$

> - 投影矩阵$P$对值应用多次线性变换和只应用一次结果相同，
	保持像不变

###	*Projection Operator*

$$\begin{array}{l}
\min & f(x) \\
s.t. & A_1 x \leq b_1 \\
& A_2 x = b_2
\end{array}$$

-	设$x^{k}$为当前迭代点，记$A_{11}$、$A_{12}$分别为紧、松
	约束，即

	$$\begin{align*}
	A_1 & = \begin{bmatrix} A_{1,1} \\ A_{1,2} \end{bmatrix},
	& b_1 & = \begin{bmatrix} b_{1,1} \\ b_{1,2} \end{bmatrix} \\
	A_{1,1} x^k & = b_{1,1}, & A_{1,2} x^k & \leq b_{1,2}
	\end{align*}$$

-	记$M = [A_{1,1}^T, A_2^T]^T$，则$s \in L_M$时是可行方向

-	对负梯度$\nabla f(x^k)$，通过$M$的投影矩阵$P_M$将其投影
	至$L_M$上即得可行下降方向$s^k = -P_M \nabla f(x^k)$

	-	$s^k \neq 0$：为$x^k$处可行下降方向
	-	$s^k = 0$：作如下讨论

####	投影方向为0

-	记$w = [u, v]^T = -(M M^T)^{-1}M \nabla f(x^k)$，则有

	$$\begin{align*}
	0 & = \nabla f(x^k) + M^T w \\
	& = \nabla f(x^k) + [A_{1,1}^T, A_2^T]
		\begin{bmatrix} u \\ v \end{bmatrix} \\
	& = \nabla f(x^k) + A_{1,1}^T u + A_2^T v
	\end{align*}$$

-	若$u \geq 0$，则$x^{k}$是KKT点

	$$
	\nabla f(x^k) + A_{1,1}^T u + A_{1,2}^T v = 0
	\Rightarrow x^k 为KKT点
	$$

-	否则若$u$中有负分量，可设$u_0 < 0$，记$\bar M$为$M$中
	去除对应列矩阵，则$\bar s^k = -P_{\bar M}\nabla f(x^k)$
	为$x^k$可行下降方向

	-	先反证法证明$\bar s^k \neq 0$，若$\bar s^k = 0$

		$$\begin{align*}
		0 & = \nabla f(x^k) - \bar M^T (\bar M \bar M^T)^{-1}
			\bar M \nabla f(x^k) \\
		& = \nabla f(x^k) + \bar M^T \beta \\
		\beta & = -(\bar M \bar M^T)^{-1} \bar M
			\nabla f(x^k)
		\end{align*}$$

		考虑到

		$$\begin{align*}
		0 & = \nabla f(x^k) + M^T w \\
		& = \nabla f(x^k) + u_0 \alpha_0 + \bar M^T \bar w
		\end{align*}$$

		> - $\alpha_0$：$M$中$u_0$对应行

		则有

		$$
		u_0 \alpha_0 + \bar M^T (\bar w - \beta) = 0
		$$

		与$M$行满秩条件矛盾，故$\bar s^k \neq 0$

	-	证明$\bar s^k$为下降方向

		$$\begin{align*}
		\nabla f(x^k)^T \bar s^k & = -\nabla f(x^k)
			P_{\bar M} \nabla f(x^k) \\
		& = -\nabla f(x^k) P_{\bar M}^T P_{\bar M}
			\nabla f(x^k) \\
		& = -\|P_{\bar M} \nabla f(x^k)\|_2^2 \leq 0
		\end{align*}$$

	-	证明$\bar s^k$方向可行（满足约束）

		-	由$P_{\bar M}$定义：$\bar M P_{\bar M} = 0$，则

			$$\begin{align*}
			\bar M \bar s^k & = -\bar M \bar P_{\bar M}
				\nabla f(x^k) \\
			& = \begin{bmatrix} \bar A_{1,1} \\ A_2
				\end{bmatrix} \bar s^k = 0
			\end{align*}$$

		-	则只需证明$\alpha_0^T \bar s^k < 0$

			$$\begin{align*}
			0 & = \nabla f(x^k) + M^T w \\
			& = \nabla f(x^k) + u_0 \alpha_0 + \bar M^T \bar w \\
			\Rightarrow & = \nabla f(x^k)^T \bar s^k + u_0
				\alpha_0^T \bar s^k + \bar w^T \bar M \bar s^k \\
			& = \nabla f(x^k)^T \bar s^k + u_0 \alpha_0^T \bar s^k
			\end{align*}$$

			考虑到$u_0 < 0$，则$\alpha_0^T \bar s^k < 0$

	> - 即此时有紧约束变为松约束

####	算法

> - 初始化：初始点$x^0$、$k=0$、精度参数$\epsilon > 0$

-	构造$M = [A_{1,1}^T, A_2^T]^T$
	-	若$M=0$（在可行域内），令$s^k = -\nabla f(x^k)$为
		迭代方向
	-	否则令$s^k = -P_M \nabla f(x^k)$为迭代方向

-	若$\|s^k\|_2^2 \geq \epsilon$
	-	若$M$为空（无可下降方向），停止
	-	若$M$非空、$u > 0$，停止
	-	否则，构建$M = \bar M$继续

-	若$\|s^k\|_2^2 > \epsilon$，确定步长$\lambda_k$

	-	显然只需保证$A_2 x_k + \lambda_k A_2 d_k \leq b_2$
		即可

	-	若$A_2 d_k < 0$，则$\lambda_k$无约束，否则

		$$
		\lambda_k = \max \{\frac {(b_2 - A_2 x_k)_i}
			{(A_2 d_k)_i}\}
		$$

	> - 即单纯型法中确定步长方法

-	得到新迭代点$x^{k+1} = x^k + \lambda_k s^k$、$k=k+1$

