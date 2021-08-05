---
title: Lagrange 对偶
categories:
  - Math Analysis
  - Optimization
tags:
  - Math
  - Analysis
  - Optimization
  - Constrained
  - Lagrange
date: 2019-06-27 01:01:10
updated: 2019-06-27 01:01:10
toc: true
mathjax: true
comments: true
description: 对偶理论
---

##	*Langrangian Duality*

拉格朗日对偶

-	考虑优化问题：找到$f(x)$满足约束的最好下界

	$$
	z^{*} = \min_{x} f(x) \\
	\begin{align*}
	s.t. \quad & g_i(x) \leq 0, i=1,2,\cdots,m \\
		& x \in X
	\end{align*}
	$$

-	考虑方程组

	$$
	\left \{ \begin{array}{l}
	f(x) < v \\
	g_i(x) \leq 0, i=1,2,\cdots,m
	\end{array} \right.
	$$

	-	**方程组无解**：$v$是优化问题的一个下界

	-	**方程组有解**：则可以推出

		$$
		\forall \lambda \geq 0, \exists x, 
		f(x) + \sum_{i=1}^m \lambda_ig_i(x) < v
		$$

		> - 显然，取$g_1 + g_2 = 0, g_1(x) > 0$是反例，不能
			推出原方程有解

	-	由以上方程组有解逆否命题：方程组无解**充分条件**如下

		$$
		\exists \lambda \geq 0,
		\min_{x} f(x) + \sum _{i=1}^m \lambda_ig_i(x) \geq v
		$$

-	由此方法推出的最好下界，即拉格朗日对偶问题

	$$
	v^{*} = \max_{\lambda \geq 0} \min_{x} f(x) +
		\sum_{i=1}^m \lambda_ig_i(x)
	$$

###	说明

-	拉格朗日对偶对实数域上的优化问题都存在，对目标函数、
	约束函数都没有要求

-	强对偶定理：$v^{*} = z^{*}$，需要$f,g$满足特定条件才成立

	-	线性规划
	-	半正定规划
	-	凸优化

	> - 即需要给约束条件加以限制，使得
		$$
		\forall \lambda \geq 0, \exists x, 
		f(x) + \sum_{i=1}^m \lambda_ig_i(x) < v
		$$
		是上述方程组有解的冲要条件

-	弱对偶定理：$v^{*} \leq z^{*}$，永远成立（以上即可证）

	-	通过弱对偶定理，可以得到原问题的一个下界
	-	对求解原问题有帮助，比如：分支界限法中快速求下界

-	对偶问题相关算法往往原问题算法在实际应用中往往更加有效

	-	*dual-simplex*
	-	*primal-dual interior point method*
	-	*augmented Lagrangian Method*

##	原始问题

约束最优化问题

$$\begin{array}{l}
\min_{x \in R^n} & f(x) \\
s.t. & c_i(x) \leq 0, i = 1,2,\cdots,k \\
& h_j(x) = 0, j = 1,2,\cdots,l
\end{array}
$$

###	*Generalized Lagrange Function*

-	引入*Generalized Lagrange Function*

	$$
	L(x, \alpha, \beta) = f(x) + \sum_{i=1}^k \alpha_i
		c_i(x) + \sum_{j=1}^l \beta_j h_j(x)
	$$

	> - $x=(x_1, x_2, \cdots, x_n) \in R^n$
	> - $\alpha_i \geq 0, \beta_j$：拉格朗日乘子

-	考虑关于x的函数

	$$
	\theta_P(x) = \max_{\alpha, \beta: \alpha_i \geq 0}
		L(x, \alpha, \beta)
	$$

	> - $P$：primal，原始问题

	-	若x满足原始问题的两组约束条件，则$\theta_P(x)=f(x)$

	-	若x违反等式约束j，取$\beta_j \rightarrow \infty$，
		则有$\theta_P(x) \rightarrow \infty$

	-	若x违反不等式约束i，取$\alpha_i \rightarrow \infty$
		，则有$\theta_P(x) \rightarrow \infty$

	则有

	$$\theta_P(x) = \left \{ \begin{array}{l}
	f(x), & x 满足原始问题约束条件 \\
	+\infty, & 其他
	\end{array} \right.$$

-	则极小化问题，称为广义拉格朗日函数的极小极大问题

	$$
	\min_x \theta_P(x) = \max_{\alpha, \beta: \alpha_i \geq 0}
		L(x, \alpha, \beta)
	$$

	与原始最优化问题等价，两问题最优值相同，记为

	$$
	p^{*} = \min_x \theta_P(x)
	$$

##	对偶问题

-	定义

	$$
	\theta_D (\alpha, \beta) = \min_x L(x, \alpha, \beta)
	$$

-	再考虑极大化$\theta_D(\alpha, \beta)$，得到广义拉格朗日
	函数的极大极小问题，即

	$$
	\max_{\alpha, \beta: \alpha \geq 0}
		\theta_D(\alpha, \beta) =
		\max_{\alpha, \beta: \alpha \geq 0} \min_x
		L(x, \alpha, \beta)
	$$

	表示为约束最优化问题如下

	$$\begin{align*}
	\max_{\alpha, \beta} & \theta_D(\alpha, \beta) =
		\max_{\alpha, \beta} \min_x L(x, \alpha, \beta) \\
	s.t. & \alpha_i \geq 0, i=1,2,\cdots,k
	\end{align*}$$

	称为原始问题的对偶问题，其最优值定义记为

	$$
	d^{*} = \max_{\alpha, \beta: \alpha \geq 0}
		\theta_D(\alpha, \beta)
	$$

##	原始、对偶问题关系

###	定理1

> - 若原始问题、对偶问题都有最优值，则
	$$
	d^{*} = \max_{\alpha, \beta: \alpha \geq 0} \min_x
		L(x, \alpha, \beta) \leq
	\min_x \max_{\alpha, \beta: \alpha \geq 0}
		L(x, \alpha, \beta) = p^{*}
	$$

-	$\forall x, \alpha, \beta$有

	$$
	\theta_D(\alpha, \beta) = \min_x L(x, \alpha, \beta)
		\leq L(x, \alpha, \beta) \leq
		\max_{\alpha, \beta: \alpha \geq 0} = \theta_P(x)
	$$

	即

	$$
	\theta_D(\alpha, \beta) \leq \theta_P(x)
	$$

-	而原始、对偶问题均有最优值，所以得证

> - 设$x^{*}$、$\alpha^{*}, \beta^{*}$分别是原始问题、对偶
	问题的可行解，且$d^{*} = p^{*}$，则其分别是原始问题、
	对偶问题的最优解








