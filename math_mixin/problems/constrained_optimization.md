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

##	凸优化一般形式

$$\mathcal{
\begin{align*}
\min_x & f(x) \\
s.t. & g_i(x) \leq 0, i=1，2,\cdots,k \\
& h_i(x) = 0, i=1,2,\cdots,l
\end{align*}
}$$

> - $f(x), g(x)$：$\mathcal{R^n}$上连续可微的凸函数
> - $h_i(x)$：$\mathcal{R^n}$上仿射函数
> - 仿射函数：满足$\mathcal{f(x)=ax+b, a \in R^n, b \in R, x \in R^n}$

##	二次规划

$$
\begin{align*}
\min_x & \mathcal{ f(x)=\frac 1 2 x^TGx + c^Tx } \\
s.t. & \mathcal{ A^x \leq b }
\end{align*}
$$

> - $\mathcal{G \in R^{n*n}}$：n阶实对称矩阵
> - $\mathcal{A \in R^{m*n}}$：m * n实矩阵
> - $\mathcal{b \in R^m}$
> - $\mathcal{c \in R^n}$

-	$\mathcal{G}$正定

	-	此时目标函数$f(x)$为凸函数
	-	凸二次规划
	-	问题有唯一全局最小值
	-	问题可可由椭球法在多项式时间内求解

-	$\mathcal{G}$半正定

	-	此时目标函数$f(x)$为凸函数
	-	半定规划
	-	若约束条件可行域不空，且目标函数在此可行域有下界，
		则问题有全局最小值

-	$\mathcal{G}$非正定

	-	目标函数有多个平稳点（局部极小），NP-hard问题

###	求解

-	椭球法
-	内点法
-	增广拉格朗日法
-	投影梯度法

##	二阶锥规划

$$\mathcal{
\begin{align*}
\min_x & f^Tx \\
s.t. & \mathcal{ \|A_ix + b_i \|_2 \leq c_i^Tx + d_i,
	i=1,2,\cdots,m } \\
	& \mathcal{ Bx=g }
\end{align*}
}$$

> - $\mathcal{f \in R^n}$
> - $\mathcal{A_i \in R^{n_i*n}}$
> - $\mathcal{b_i \in R^{n_i}}$
> - $\mathcal{c_i \in R^{n_i}}$
> - $\mathcal{d_i \in R}$
> - $\mathcal{B \in R^{p*n}}$
> - $\mathcal{g \in R^n}$


-	二阶锥规划可以使用内点法很快求解（多项式时间）

-	$\mathcal{A_i=0,i=1,\cdots,m}$：退化为LP

-	一般的二阶规划可以转换为二阶锥规划

	$$\mathcal{
	X^TAX + qTX + C \leq 0 \Rightarrow \\
	\|A^{1/2}x + \frac 1 2 A^{-1/2}q\|^{1/2} \leq
		-\frac 1 4 q^TA^{-1}q - c
	}$$





