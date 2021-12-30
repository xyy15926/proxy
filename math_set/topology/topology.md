---
title: 拓扑空间
categories:
  - Set
  - Topology
tags:
  - Math
  - Set
  - Math Space
  - Topolopy
date: 2021-12-30 16:20:12
updated: 2021-12-30 16:20:12
toc: true
mathjax: true
description: 
---

##	*Topological Space*

-	*Topological Space* 拓扑空间 $(X, \tau)$
	-	其中
		-	集合 $X$：其中元素称为拓扑空间 $(X, \tau)$ 的点
		-	拓扑结构 $\tau$：涵盖开集、闭集、领域、开核、闭包、导集、滤子等概念
	-	在拓扑空间上可以形式化的定义收敛、联通、连续等概念

> - <https://zh.wikipedia.org/wiki/%E6%8B%93%E6%89%91%E7%A9%BA%E9%97%B4>

###	拓扑空间部分公理

####	开集公理

-	开集公理：$X$ 的子集的集合族 $D$ 称为开集系（其中元素称为开集），当前仅当其满足如下开集公理
	-	$o_1$：$\emptyset \in D, X \in D$
	-	$o_2$：若 $A_{\lambda} \in D, \lambda \in \Lambda$，则 $\bigcup_{\lambda \in \Lambda} A_{\lambda} \in D$（对任意并运算封闭）
	-	$o_3$：若 $A,B \in D$，则 $A \bigcap B \in D$（对有限交运算封闭）

-	由开集出发定义
	-	闭集：$X$ 的子集 $A$ 是闭集，当前仅当 $X-A$ 是开集
	-	邻域：$X$ 的子集 $U$ 是 $x$ 的邻域，当前仅当存在开集 $O$，使 $x \in O \subseteq U$
	-	开核：$X$ 的子集 $A$ 的开核 $A^{\circ}$ 等于 $A$ 包含的所有开集之并

> - 全体开集决定了空间的拓扑性质

####	闭集公理

-	闭集公理：$X$ 的子集的集合族 $F$ 称为闭集系（其中元素称为闭集），当前仅当其满足如下闭集公里
	-	$c_1$：$\emptyset \in F, X \in F$
	-	$c_2$：若 $A_{\lambda} \in F, \lambda \in \Lambda$，则 $\bigcap_{\lambda \in \Lambda} A_{\lambda} \in F$（对任意交运算封闭）
	-	$c_3$：若 $A,B \in F$，则 $A \bigcup B \in F$（对有限闭运算封闭）

-	由闭集出发定义
	-	开集：$X$ 的子集 $A$ 是开集，当前仅当 $X-A$ 是闭集
	-	闭包：$X$ 的子集 $A$ 的闭包 $\bar A$ 等于包含 $A$ 的所有闭集之交

####	邻域公理

-	邻域公理：$X$ 的映射 $U: X \rightarrow P(P(X))$（$P(X)$ 为幂集），将 $x \in X$ 映射至子集族 $U(x)$，$U(x)$ 称为 $x$ 的领域系（其中元素称为 $x$ 的领域），当且仅当 $\forall x \in X$，$U(x)$ 满足
	-	$U_1$：若 $U \in U(x)$，则 $x \in U$
	-	$U_2$：若 $U,V \in U(x)$，则 $U \bigcap V \in U(x)$（对有限交封闭）
	-	$U_3$：若 $U \in U(x), U \subseteq V \subseteq X$，则 $V \in U(x)$
	-	$U_4$：若 $U \in U(x)$，则 $\exists V \in U(x), V \subseteq U, \forall y \in V, U \in U(y)$

-	从邻域出发定义
	-	开集：$X$ 的子集 $O$ 是开集，当前仅当 $\forall x \in O, O \in U(x)$
	-	开核：$X$ 的子集 $A$ 的开核 $A^{\circ} = \{x| \exists U \in U(x), U \subseteq A \}$
	-	闭包：$X$ 的子集 $A$ 的闭包 $\bar A = \{ x| \forall U \in U(x), U \bigcap A \neq \emptyset \}$

####	闭包公理

-	闭包公理：$X$ 的幂集 $P(X)$ 上的一元运算 $c: P(x) \rightarrow P(x)$ 称为闭包运算当且仅当运算 $c$ 满足
	-	$A_1$：$A \subseteq c(A)$
	-	$A_2$：$c(c(A)) = c(A)$
	-	$A_3$：$c(A \bigcup B) = c(A) \bigcup c(B)$
	-	$A_4$：$c(\emptyset) = \emptyset$

-	由闭包出发定义
	-	闭集：$X$ 的子集 $A$ 是闭集，当前仅当 $A = \bar A$
	-	开核：$X$ 的子集 $A$ 的开核 $A^{\circ} = X - \overline {X - A}$
	-	邻域：$X$ 的子集 $U$ 是点 $x$ 的邻域，当且仅当 $x \notin \overline {X-U}$

####	开核公理

-	开核公理：$X$ 的幂集 $P(X)$ 上的一元运算 $o:P(X) \rightarrow P(X)$ 称为开核运算，当且仅当运算 $o$ 满足
	-	$l_1$：$o(A) \subseteq A$
	-	$l_2$：$o(o(A)) = o(A)$
	-	$l_3$：$o(A \hat B) = o(A) \hat o(B)$
	-	$l_4$：$o(X) = X$

-	由开核出发定义
	-	开集：$X$ 的子集 $A$ 是开集，当且仅当 $A = A^{\circ}$
	-	邻域：$X$ 的子集 $U$ 是点 $x$ 的邻域，当且仅当 $x \in U^{\circ}$
	-	闭包：$X$ 的子集 $A$ 的闭包 $\bar A = X - (X-A)^{\circ}$

####	导集公理

-	导集公理：$X$ 的幂集 $P(X)$ 上的一元运算 $d:P(X) \rightarrow P(X)$ 称为导集运算，当且仅当
	-	$D_1$：$d(\emptyset) = \emptyset$
	-	$D_2$：$d(d(A)) \subseteq d(A) \bigcup A$
	-	$D_3$：$\forall x \in X, d(A) = d(A - \{x\})$
	-	$D_4$：$d(A \bigcup B) = d(A) \bigcup d(B)$

-	由导集出发定义
	-	闭集：$X$ 的子集 $A$ 是闭集，当且仅当 $d(A) \subseteq A$

####	其他一些结论

-	$X$ 是不连通空间当且仅当 $X$ 中存在既开又闭得非空真子集

###	拓扑空间性质

-	全集 $X$ 之间可以拥有不同的拓扑（空间），形成偏序关系
	-	当拓扑 $T_1$ 的每个开集都是拓扑 $T_2$ 的开集时，称 $T_2$ 比 $T_1$ 更细、$T_2$ 比 $T_1$ 更粗
	-	仅依赖特定开集存在而成立结论，在更细的拓扑上成立；类似的，依赖特定集合不是开集成立的结论，在更粗的拓扑上也成立
	-	最粗的拓扑是由空集、全集两个元素构成的拓扑；最细的拓扑是离散拓扑

-	连续映射：拓扑空间上的映射 $f$ 称为连续映射，当且仅当其满足以下条件之一
	-	$f$ 对任何开集的原象是开集
	-	$f$ 对任何闭集的原象是闭集
	-	对点 $f(x)$ 的任一邻域 $V$，对存在点 $x$ 的邻域，使得 $f(U) \subset V$，则称 $f(x)$ 在 $x$ 处连续，连续映射即在所有点连续的映射
	-	对任一集合 $A$，$f(\bar A) \subseteq \overline{f(A)}$ 成立
	-	对任一集合 $A$，$f^{-1}(A^{\circ}) \subseteq (f^{-1}(A))^{\circ}$ 成立

-	同胚映射：两个拓扑空间间的连续双射
	-	存在同胚映射的两个空间称为同胚的，在拓扑学观点上，同胚的空间等同

###	一些概念

-	给定拓扑空间 $(X, \tau)$，$A \subseteq X$，可以定义
	-	内部：$A$ 的开核 $A^{\circ}$，其中点即为内点
	-	外部：$A$ 的闭包补集 $X - \bar A$，其中点即为外点
	-	边界：$\bar A \cap \overline {X-A}$，其中点即为边界点
	-	触点：$A$ 的闭包 $\bar A$ 中点
	-	稠密性/稠密集：当且仅当 $\bar A = X$ 时，称 $A$ 在 $X$ 中是稠密的
	-	边缘集：当且仅当 $X-A$ 在 $A$ 中稠密时，称 $A$ 时 $X$ 的边缘集
	-	疏性/疏集：当且仅当 $\bar A$ 是 $X$ 中边缘集时，称 $A$ 是 $X$ 中疏集
	-	第一范畴集：当且仅当 $A$ 可以表示为可数个疏集并时，称 $A$ 为 $X$ 中第一范畴集；不属于第一范畴集则为第二范畴集
	-	聚点：当且仅当 $x \in \overline {A-\{x\}}$ 时，$x$ 称为 $A$ 的聚点（即 $x$ 的任意邻域至少包含 $x$ 以外的 $A$ 的一个点）
	-	导集：$A$ 的所有聚点组成的集合
	-	孤立点：不是 $A$ 的聚点的 $A$ 中的点
	-	孤点集/离散集：所有点都是孤立点
	-	自密集：所有点都是聚点
	-	自密核：最大自密子集
	-	无核集：自密核为 $\emptyset$（即任意非空子集都含有孤立点）
	-	完备集：导集为自身
		-	即集合自身是没有空隙的

-	*First-countable Space* 第一可数空间：$\forall x \in X, \exists U_1,U_2,\cdots$ 使得对任意领域 $V$，$exists i \in N, U_i \subseteq V$
	-	即有可数的领域基的拓扑空间
	-	大部分常见空间为第一可数的，所有度量空间均可数

###	紧致性

-	紧致性：拓扑空间 $X$ 是紧致的，若对于任意由 $X$ 开子集构成的集合族 $C$ 使得 $X = \bigcup_{x \in C} x$，总存在有限子集 $F \subseteq C$，使得 $X = \bigcup_{x \in F} x$
	-	即拓扑空间的所有开覆盖都有有限子覆盖

-	紧致性蕴含
	-	紧致性是有限性之后最好的事情
	-	很多容易通过有限集合证明的结果，通过较小的改动即可转移至紧致空间上

###	稠密性

-	稠密：给定拓扑空间 $X$，对 $A \subseteq X$，若 $\forall x \in X$，$x$ 的任一邻域与 $A$ 交集不空，称 $A$ 在 $X$ 中稠密
	-	$A$ 在 $X$ 中稠密当且仅当以下之一成立
		-	唯一包含 $A$ 的闭集为 $X$
		-	$A$ 的闭包为 $X$
		-	$A$ 的补集内部是空集

-	直观上
	-	若 $X$ 中任一点可被 $A$ 中点很好逼近，则称 $A$ 在 $X$ 中稠密

