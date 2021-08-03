---
title: 集合
categories:
  - Set
tags:
  - Math
  - Set
  - Order
  - Base
date: 2019-07-21 00:46:35
updated: 2021-07-19 09:19:48
toc: true
mathjax: true
comments: true
description: 代数运算
---

##	集合

###	势

> - 等势：若集合 $X, Y$ 之间存在双射 $\phi: X \rightarrow Y$，则称 $X, Y$ 等势
> - 可数/可列集合：与自然数集合、其子集等势的集合称为可数集合，否则称为不可数集合

-	等势构成集合之间的等价关系
	-	集合 $X$ 的等势类记为 $|X|$
	-	若存在单射 $\phi: X \rightarrow Y$，则记为 $|X| \leq |Y|$
-	一些基本结论
	-	自然数集 $N = \{0, 1, 2, 3, \cdots\}$ 和闭区间 $[0,1]$ 不等势

> - <https://zhuanlan.zhihu.com/p/34097692>

##	序

> - 偏序集：若集合 $A$ 上有二元关系 $\leq$ 满足以下性质，则称集合 $A$ 为偏序集，关系 $\leq$ 称为偏序关系
> > -	反身性：$\forall x \in A, x \leq x$
> > -	传递性：$(x \leq y) \wedge (y \leq z) \Rightarrow x \leq z$
> > -	反称性：$(x \leq y) \wedge (y \leq x) \Rightarrow x = y$

> - 全序集：若 $\leq$ 是集合上的偏序关系，若对每个$x, y \in A$，必有 $x\leq y$ 或 $y \leq x$，则称集合 $A$ 为全序集，关系 $\leq$ 为全序关系

> - 良序集：若集合 $A$ 每个自己都有极小元，则称为良序集

-	特点
	-	偏序指集合中只有部分成员之间可比较
	-	全序指集合全体成员之间均可比较
	-	良序集则是不存在无穷降链的全序集（可有无穷升链）

###	序数

> - 序数：若集合 $A$ 中每个元素都是 $A$ 的子集，则称 $A$ 是传递的。而 $A$ 对于关系 $\in$ 构成良序集，则称 $A$ 为序数

-	满足如下形式的集合即为序数

	$$
	\{\phi, \{\phi\}, \{\phi, \{\phi\}\}, \{\phi, \{\phi\}, \{\phi, \{\phi\}\}\} \}, \cdots
	$$

-	序数的性质（引理）
	-	若 $\alpha$ 为序数，$\beta \in \alpha$，则 $\beta$ 也是序数
	-	对任意序数 $\alpha, \beta$，若 $\alpha \subset \beta$，则 $\alpha \in \beta$
	-	对任意序数 $\alpha, \beta$，必有 $\alpha \subseteq \beta$ 或 $\beta \subseteq \alpha$

-	由以上，序数性质的解释
	-	序数是唯一的，都满足上述形式
	-	序数都是由自己之前的所有序数构造而来
	-	对任意序数 $\alpha$，有 $\alpha = \{\beta: \beta < \alpha \}$ （$ < $ 表示偏序关系）

-	将 $0, 1, 2, \cdots$ 依次对应上述序数，即给出自然数和序数

	$$
	0 := \phi, 1 := \{phi\}, 2 := \{\phi, \{phi\}\}, \cdots
	$$

> - <https://zh.wikipedia.org/wiki/%E5%BA%8F%E6%95%B0>
> - 自然数可用于描述集合大小（势，基数）、序列中元素的位置（序，序数）

####	良序定理

> - *Zermelo* 良序定理：任何集合 $P$ 都能被赋予良序

-	*Zermelo* 良序定理和 *ZFC* 选择公理等价，可以由选择公理证明
	-	由选择公理，可以一直从集合中选择元素，建立偏序关系
	-	而集合有限，则集合和序数之间可以建立双射

##	基

###	基数

> - 基数：序数 $k$ 为基数，若对任意序数 $\lambda < k$，都有 $|\lambda| < |k|$
> - *Counter* 定理：设 $A$ 为集合，$P(A)$ 为 $A$ 的幂集，则有 $|A| \leq |P(A)|$

-	基数是集合势的标尺

-	数的集合的基数
	-	自然数集合基数 $\aleph_0$：最小的无限基数
	-	实数集集合基数称为 *continuum* 连续统

-	连续统假设：不存在一个集合，基数在自然数集和连续统之间
	-	哥德尔证明：连续统假设与公理化集合论体系 *Zermelo-Fraenkel set theory with the axiom of choice* 中不矛，即不能再 *ZFC* 中被证伪
	-	科恩证明：连续统假设和 *ZFC* 彼此独立，不能在 *ZFC* 公理体系内证明、证伪

> - <https://zh.wikipedia.org/wiki/%E5%9F%BA%E6%95%B0_(%E6%95%B0%E5%AD%A6)>






