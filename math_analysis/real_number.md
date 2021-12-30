---
title: 实数（度量）空间
categories:
  - Math Analysis
  - Real Analysis
tags:
  - Math
  - Analysis
  - Real Analysis
  - Real Number
date: 2021-12-30 16:27:14
updated: 2021-12-30 16:27:45
toc: true
mathjax: true
description: 
---

##	实数定义

-	实数定义：若 $R$ 完备，且 $(R,+,\times)$ 是有序域，则 $(R,+,\times)$ 是实数域，$R$ 是实数集，$R$ 中元素即为实数
	-	完备的有序域同构均同构于 $R$

-	数系的扩展
	-	自然数系到有理数系的扩展基于代数运算的需求
	-	有理数系到实数系的扩展是拓扑学的需求
		-	为代数体系赋予形状，定义远近、长短
		-	是建立几何和分析结构的基础
		-	有理数在实数中的稠密

-	实数 $R$ 即为有理数 $Q$ 建立在 **绝对值度量** 上的完备化
	-	与日常欧氏距离概念吻合，符合直观经验，因此实数是描述现实世界的有力工具

	> - 奥斯洛夫斯基定理：任何非平凡的有理数 $Q$ 的绝对赋值要么等价于通常实数域的绝对赋值，要么等价于 *p-进数* 的绝对赋值

###	*Dedekind Property*

-	*Dedekind Completeness* 戴德金原理（以下两种表述等价），即全序集 $R$ 具有完备性的定义
	-	表述 1：$\forall A, B \subseteq R, A \neq \emptyset, B \neq \emptyset$，若 $\forall x \in A, y \in B$ 总有 $x \leq y$，则 $\forall a \in A, b \in B$ 有 $\exists c \in R, a \leq c \leq b$
	-	表述 2：若 $A,B \subseteq R$ 满足如下性质，则 $\forall a \in A, b \in B$ 有 $\exists c \in R, a \leq c \leq b$
		-	$A \neq \emptyset, B \neq \emptyset$
		-	$A \bigcup B = R$
		-	$\forall x \in A, y \in B$ 总有 $x < y$

-	戴德金原理的蕴含
	-	在数轴上任意选择一点切分，一定恰好切到实数
	-	其中 $a \leq c \leq b$ 中包含了确界
	-	戴德金原理在此即作为实数的根本性质，*Rudin* 书中则以上确界原理作为实数根本性质

-	实数完备性（即戴德金原理）蕴含阿基米德性质（以下表述等价）
	-	$\forall x \in R^{+}$ 有 $\exists n \in Z^{+}, nx > 1$
	-	$\forall x \in R^{+}, \forall y \in R^{+}$ 有 $\exists n \in Z^{+}, nx > y$
	-	$\forall y \in R$ 有 $\exists n \in Z^{+}, n > y$
	-	$\lim_{n \rightarrow \infty} \frac 1 n = 0$，即 $\forall \epsilon \in R^{+}$ 有 $\exists N \in R^{+}, \forall n > N, \frac 1 n < \epsilon$
	-	$\lim_{n \rightarrow \infty} \frac 1 {2^n} = 0$，即 $\forall \epsilon \in R^{+}$ 有 $\exists N \in R^{+}, \forall n > N, \frac 1 {2^n} < \epsilon$

-	阿基米德性质蕴含
	-	实数域中没有无穷大、无穷小的元素

###	实数（空间）完备性定理

-	上确界定理：$\forall S \in R, S \neq \emptyset$，若 $S$ 在 $R$ 内有上界，则 $S$ 在 $R$ 内有上确界

-	区间套定理：若数列 $\{a_n\}$、$\{b_n\}$ 满足条件：$\forall n \in Z^{+}, a_n \leq a_{n+1} \leq b_{n+1} \leq b_n$，$\lim_{n \rightarrow \infty} b_n - a_n = 0$，则有：$\lim_{n \rightarrow \infty} a_n = \lim_{b_n \rightarrow \infty b_n = c$，且$c$ 是在实数空间唯一

-	单调有界定理：单调有界数列收敛
-	有限覆盖定理：若开区间集 $E$ 覆盖闭区间 $[a,b]$，总可以从 $E$ 中选取有限个开区间，使其覆盖 $[a,b]$
-	聚点定理：$R$ 上无穷、有界的子集 $S$ 至少有一个聚点
-	柯西收敛准则：实数域上柯西列收敛
-	致密性定理：任意有界数列有收敛子列

> - 其中区间套定理、柯西收敛准则本身不内蕴实数域的无限性，需要补充阿基米德性质才足够强
> - <https://zhuanlan.zhihu.com/p/48859870>

