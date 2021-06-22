---
title: Norm
tags:
  - 分析
  - 泛函分析
categories:
  - 分析
  - 泛函分析
date: 2019-07-21 00:46:35
updated: 2019-07-21 00:46:35
toc: true
mathjax: true
comments: true
description: Norm
---

##	*Norm*

###	$\mathcal{L_p}$ 范数

-	$\mathcal{L_p}$ 范数

	$$
	\|x\|_p = (|x_1|^p + |x_2|^p + \cdots + |x_N|^p)^{1/p}
	$$

	> - $x \in R^n$

-	$\mathcal{L_p}$ *Dual Norm* 对偶范数

	$$
	\|x\|^{*} = \mathop \sup_{z}{x^Tz|\|z\| \leq 1}
	$$

	> - $x \in R^N$
	> - $\|x\|$：$x$的某个范数

####	Holder定理

$\|x\|^{*}_p = \|x\|_q$

> - $\frac 1 p + \frac 1 q = 1$

###	*P-stable Distribution*

> - p稳定分布：若$R$上的分布$D$，若$\exists p \geq 0$使得对
	任意n个实数$v_1,v_2,\cdots,v_n$、独立同分布$D$的n个随机
	变量$X_1,X_2,\cdots,X_n$，满足随机变量$\sum_i v_iX_i$
	同随机变量$(\sum_i |v_i|^p)^{1/p} X$分布相同

-	有解析解的稳定分布仅有

	-	$p=1$：柯西分布

		$$
		c(x) = \frac 1 {\pi} \frac 1 {1 + x^2}
		$$

	-	$p=2$：高斯分布

		$$
		g(x) = \frac 1 {\sqrt {2\pi}} e^{-\frac {x^2} 2}
		$$

-	稳定分布将$L_p$范数$\|v\|_p$同内积联系，可以用于

	-	估计$L_p$f距离：$v = v^{(1)} - v^{(2)}$



