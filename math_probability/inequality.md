---
title: 概率不等式
categories:
  - Probability
tags:
  - Math
  - Probability
  - Uncharted
  - Inequality
date: 2019-03-26 00:04:19
updated: 2021-07-19 15:50:35
toc: true
mathjax: true
comments: true
description: 参考
---

##	Inequality

###	*Azuma-Hoeffding Inequality*

*Azuma-Hoeffding* 不等式：设 ${X_i:i=0,1,2,\cdots}$ 是鞅差序列，且 $|X_k - X_{k-1}| < c_k$，则

$$\begin{align*}
super-martingale:
P(X_N - X_0 \geq t) \leq exp \left ( \frac {-t^2}
	{2\sum^N_{k=1} c_k^2} \right ) \\

sub-martingale:
P(X_N - X_0 \leq -t) \leq exp \left ( \frac {-t^2}
	{2\sum^N_{k=1} c_k^2} \right ) \\

martingale:
P(|X_N - X_0| \geq t) \leq exp \left ( \frac {-t^2}
	{2\sum^N_{k=1} c_k^2} \right )
\end{align*}$$

###	*Hoeffding Inequality*

*Hoeffding* 不等式：考虑随机变量序列 $X_1, X_2, \cdots, X_N, X_i \in [a_i, b_i]$

-	对随机变量 $\bar X = \frac 1 N \sum_{i=1}^N {X_i}$，对任意 $t>0$ 满足

	$$\begin{align*}
	P(\bar X - E \bar X \geq t) \leq exp(\frac {-2N^2t^2}
		{\sum_{i=1}^N (b_i - a_i)^2} ) \\
	P(E \bar X - \bar X \geq t) \leq exp(\frac {-2N^2t^2}
		{\sum_{i=1}^N (b_i - a_i)^2} ) \\
	\end{align*}$$

-	对随机变量 $S_N = \sum_{i=1}^N X_i$，对任意 $t>0$ 满足

	$$\begin{align*}
	P(S_N - E S_N \geqslant t) & \leqslant exp \left (
		\frac {-2t^2} {\sum_{i=1}^n (b_i - a_i)^2} \right ) \\
	P(E S_N - S_N \geqslant t) & \leqslant exp \left (
		\frac {-2t^2} {\sum_{i=1}^n (b_i - a_i)^2} \right )  \\
	\end{align*}$$

> - 两不等式可用绝对值合并，但将不够精确

###	*Bretagnolle-Huber-Carol Inequility*

*Bretagnolle-Huber-Carol* 不等式：${X_i: i=1,2,\cdots,N} i.i.d. M(p1, p_2, \cdots, p_k)$ 服从类别为 $k$ 的多项分布

$$
p{\sum_{i=1}^k |N_i - Np_i| \geq \epsilon} \leq
	2^k exp \left ( \frac {- n\epsilon^2} 2  \right )
$$

> - $N_i$：第 $i$ 类实际个数

