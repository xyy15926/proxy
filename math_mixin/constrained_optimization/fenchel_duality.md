---
title: Fenchel-Legendre Duality
tags:
  - 最优化
  - 约束优化
categories:
  - 最优化
  - 约束优化
date: 2019-07-21 00:46:35
updated: 2019-07-21 00:46:35
toc: true
mathjax: true
comments: true
description: Fenchel-Legendre Duality
---

##	*Legendre Transformation*

勒让德变换：用 $f^{ * }(p)$ 表示凸、可导函数 $f(x)$ 的变换，其中 $p$ 是 $f(x)$ 导数

$$
f^{*}(p) = p^T x - f(x)|_{\frac {d(p^T x - f(x))} {dx} = 0}
$$

> - $x$：参数，满足 $\frac {d(p^T x - f(x))} {dx} = 0$，随 $p$ 取值改变
> - 可导：有导数；凸：导数唯一

-	勒让德变换是实变量的实值凸函数的对合变换
	-	把定义在线性空间上的函数变换至对偶空间的函数
	-	是点、（切）线之间对偶关系的应用
		-	严格凸函数中，切线、导数一一对应
		-	函数关系 $f(x)$ 可使用 $(x, y=f(x))$ 点集表示，也可用**切线集合**表示

> - *involution* 对合：对合函数 $f$ 的反函数的为自身，即 $f(f(x))=x$；对合线性变换 $V$ 满足 $V^2 = E$

###	*Legendre* 变换理解（按 *Fenchel* 共轭）

-	$f^{*}(p)$：可理解为斜率为 $p$、同 $f(x)$ 有交点 $x_0$ 的直线在零点处值（截距）和 $f(x_0)$ 的最大差

	![fenchel_conjugate_max_interception](imgs/fenchel_conjugate_max_interception.png)

-	$x$：可以理解为函数 $f(x)$ 上距离给定斜率为 $p$、过原点的直线 $f(x)=px$ 竖直距离最大的点

	![fenchel_conjugate_max_vertical_distance](imgs/fenchel_conjugate_max_vertical_distance.png)

	> - 类似一个端点为 $0$ 的 *Bregman* 散度

-	*Legendre* 变换为对合变换，进行两次的变换得到原函数

	![fenchel_conjugate_transformation_cycle](imgs/fenchel_conjugate_transformation_cycle.png)

	$$\begin{align*}
	f^{**}(x) & = \sup_{p \in dom(f^{*})} [x^T p - f^{*}(p)] \\
	& = \sup_{u \in dom(f)}[x^T \nabla f(u) -
		\nabla f(u)^T u + f(u)] \\
	& = \sup_{u \in dom(f)}[f(u) + \nabla f(u)^T (x-u)] \\
	& = f(x)
	\end{align*}$$

-	若视凸函数 $f(x)$ 视为积分，则其共轭 $f^{ * }(x)$ 为对另一轴积分，二者导函数互为反函数

	$$
	f(x) + f^{*}(p) = xp, p = \frac {df(x)} {dx}
	$$

> - 以上性质均按 *Fenchel* 共轭，但要求 $f(x)$ 为凸、可导函数，故等价于 *Legendre* 变换

###	*Legendre* 变换最大值式定义

$$\begin{align*}
L(p, x) &= px - f(x) \\
\frac {\partial (px - f(x))} {\partial x} &= p - \frac {df(x)} {dx} = 0 \\
\Rightarrow & p = \frac {df(x)} {dx}
\end{align*}$$

-	*Legendre* 变换可以视为寻找 $px-f(x)$ 最大值（如前述）
	-	$f(x)$ 为凸函数，则 $p=\frac {df(x)} {dx}$ 是最大值点
	-	则将 $f(x)$ 导函数的反函数 $x=f^{-1}(p)$ 带入即可

###	*Legendre* 变换数学性质

-	标度性质

	$$\begin{align*}
	f(x) & = a g(x) \rightarrow f^{*}(p) = a g^{*}(\frac p a) \\
	f(x) & = g(ax) \rightarrow f^{*}(p) = g^{*}(\frac p a)
	\end{align*}$$

	由此，$r$次齐次函数的勒让德变换是$s$次齐次函数，满足

	$$
	\frac 1 r + \frac 1 s = s
	$$

-	平移性质

	$$\begin{align*}
	f(x) & = g(x) + b \rightarrow f^{*}(p) = g^{*}(p) - b
	f(x) & = g(x+y) \rightarrow f*^{*}(p) = g^{*}(p) - py
	\end{align*}$$

-	反演性质

	$$
	f(x) = g^{-1}(x) \rightarrow f^{*}(p) = -p g^{*}(\frac 1 p)
	$$

-	线性变换性质

	$$
	(Af)^{*} = f^{*}A^{*}
	$$

	> - $f$：$R^n$上的凸函数
	> - $A$：$R^n \rightarrow R^m$的线性变换
	> - $A^{*}: <Ax, y^{*}> = <x, A^{*}y^{*}>$：$A$伴随算子

##	*Fenchel Conjugate* / 凸共轭

$$
f^{*}(p) = \sup_{x \in R}{p^Tx - f(x)}
$$

-	*Fenchel* 共轭是对 *Legendre* 变换的扩展，不再局限于凸、可导函数
	-	*Fenchel* 共轭可类似 *Legendre* 理解，但是适用范围更广
	-	对凸函数 *Fenchel* 共轭的共轭即为原函数，对非凸函数 *Fenchel* 共轭得到**原函数凸包**
	-	用罗尔中值定理描述极值、导数关系：兼容 *Legendre* 变换中导数支撑面

> - 非凸函数线性外包络是凸函数

###	*Fenchel-Young*不等式

$$
f(x) + f^{*}(p) \geq <p, x>
$$

-	证明

	$$\begin{align*}
	f(x) + f^{*}(p) & = f(x) + \sup_{x \in dom(f)} {(x^T p - f(x))} \\
	& \geq f(x) + x^T p - f(x) = x^T p
	\end{align*}$$

-	按积分理解，仅$p$为$x$共轭时取等号

	![fenchel_conjugate_integration_for_fenchel_young_ineq](imgs/fenchel_conjugate_integration_for_fenchel_young_ineq.png)

###	*Fenchel Conjugate* 推导 *Lagrange Duality*

-	原问题 *Prime* 

	$$\begin{align*}
	& \min {f(x)} \\
	s.t. & g(x) \leq 0 \\
	\end{align*}$$

-	约束条件 $g(x) \leq 0$ 扰动函数化、求 *Fenchel* 共轭

	$$\begin{align*}
	p(u) & = \inf_{x \in X, g(x) \leq u} f(x) \\
	p^{*}(y) & = \sup_{y \in R^r} \{u^T y - p(u)\}
	\end{align*}$$

-	记 $\lambda = -y$，并将 $y=-\lambda$ 带入 $-p^{*}(y)$ 中得到

	$$\begin{align*}
	-p^{*}(y) & = \inf_{u \in R^r} \{p(u) - u^T y\} \\
	d(\lambda) & = \inf_{u \in R^r} \{p(u) + u^T \lambda\} \\
	& = \inf_{u \in R^r} \{\inf_{x \in X, g(x) \leq u} f(x)
		+ \lambda^T u\}
	\end{align*}$$

	> - $\lambda = -y$

-	将 $\inf_{x \in X, g(x) \leq u}$ 外提，并考虑到约束 $g(x) \leq u$（即 $u \geq g(x)$），则

	$$\begin{align*}
	\lambda \geq 0 & \Rightarrow \lambda^T g(x) \leq \lambda u \\
	d(\lambda) & = \left \{ \begin{array}{l}
			\inf_{x \in X} \{f(x) + \lambda^T g(x)\},
				& \lambda \geq 0 \\
			-\infty, & otherwise
		\end{array} \right.
	\end{align*}$$

-	考虑 *Fenchel* 不等式

	$$\begin{align*}
	p(u) + p^{*}(-y) & \geq u^T (-y) \\
	p(0) + p^{*}(-y) & \geq 0 \\
	p(0) & \geq -p^{*}(-y) \\
	p(0) & \geq d(\lambda)
	\end{align*}$$

-	则可得 *Lagrange* 对偶 *Prime*、*Dual* 最优关系

	$$
	L(x, \lambda) = f(x) + \lambda^T g(x), \lambda \geq 0 \\
	D^{*} := \max_{\lambda \geq 0} \min_x L(x, \lambda) \leq
		\min_x \max_{\lambda \geq 0} L(x, \lambda) =: P^{*}
	$$

	![fenchel_conjugate_dual_gap](imgs/fenchel_conjugate_dual_gap.png)

###	*Lagrange Duality* 推导 *Fenchel* 对偶

> - *Fenchel* 对偶可以视为 *Lagrange* 对偶的应用

-	原问题、等价问题

	$$\begin{align*}
	& \min_x & f(x) - g(x) \\
	\Leftrightarrow & \min_{x,z} & f(x) - g(z) \\
	& s.t. & x = z
	\end{align*}$$

-	对上式取 *Lagrange* 对偶 $L(u)$、等价得到

	$$\begin{align*}
	L(u) &= \min_{x,z} f(x) - g(z) + u^T(z-x) \\
	&= -(f^{*}(u) - g^{(-u)})
	\end{align*}$$

![fenchel_conjugate_duality](imgs/fenchel_conjugate_duality.png)

> - *Fenchel* 对偶：寻找截距差值最大的平行切线


