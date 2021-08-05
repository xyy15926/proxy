---
title: 凸分析
categories:
  - Math Analysis
  -	Real Analysis
tags:
  - Math
  - Analysis
  - Real Analysis
  - Convex
date: 2019-07-21 00:46:35
updated: 2019-07-21 00:46:35
toc: true
mathjax: true
comments: true
description: 凸分析
---

##	Notations and Terminology

###	*Strong Convexity*

-	凸函数 $f$ 满足

	$$
	\forall x, y \in R, \forall \lambda \in (0,1), 
		f(\lambda x + (1-\lambda) y) \leq \lambda f(x) +
		(1-\lambda)f(y)
	$$

-	强凸函数：不等式严格不等的凸函数
	-	为保证强凸性，常添加二次项保证，如：增广拉格朗日

###	凸集相关标记

> - $C \subseteq R^N$：非空凸集
> - $x \in R^N$

-	点 $x \in R^N$ 到 $C$ 的距离为

	$$D_C(x) = \min_{y \in C} \|x-y\|_2$$

-	点 $x \in R^N$ 在 $C$ 上投影为

	$$P_Cx \in C, D_C(x) = \|x - P_Cx\|_2$$

	-	$C \subseteq R^N$：闭凸集

-	*Indicator Function* 凸集 $C$ 的示性函数为

	$$
	l_C(x) = \left \{ \begin{array}
		0 & if x \in C \\
		+\infty & if x \notin C
	\end{array} \right.
	$$





