---
title: 环
categories:
  - Algebra
  - Abstract Algebra
  - Number Theory
tags:
  - Math
  - Algebra
  - Abstract Algebra
  - Number Theory
  - Ring
date: 2021-09-06 11:19:01
updated: 2021-10-20 16:47:23
toc: true
mathjax: true
description: 
---

##	*Ring*

-	环：集合 $R$ 和定义与其上的二元运算 $+$、$\cdot$ 构成的三元组 $(R, +, \cdot)$，若它们满足
	-	$(R, +)$ 形成一个交换群，其单位元称为零元，记作 $0$
	-	$(R, \cdot)$ 形成一个半群
	-	$\cdot$ 乘法关于 $+$ 加法满足分配律
		-	乘法优先级高于加法

> - 实际中，运算符号被省略，称 $R$ 为环

###	环的衍生

||（除 0 元）封闭|单位元|逆元|交换|
|-----|-----|-----|-----|-----|
|交换环|-|-|-|是|
|幺环|-|是|-|-|
|无零因子环|是|-|-|-|
|整环|是|是|-|是|
|除环|是|是|是|-|
|体|是|是|是|是|

-	交换环：若环 $(R, +, \cdot)$ 中 $(R, \cdot)$ 满足交换律构成交换半群，则称 $R$ 为交换环

-	幺环：若环 $(R, +, \cdot)$ 中 $(R, \cdot)$ 构成幺半群，则称 $R$ 为幺环
	-	此时幺半群 $(R, \cdot)$ 的幺元 $1$ 也成为环的幺元

-	无零因子环：若环 $R$ 中没有非 0 的零因子，则称 $R$ 为无零因子环
	-	此定义等价于以下任意
		-	$R \setminus \{0\}$ 对乘法形成半群
		-	$R \setminus \{0\}$ 对乘法封闭
		-	$R$ 中非 0 元素的乘积非 0

-	整环：无零因子的交换幺环
	-	例
		-	整数环
		-	多项式环

-	除环：若 $R$ 为幺环，且 $R \setminus \{0\}$ 对 $R$ 上乘法形成群，则 $R$ 称为除环
	-	除环不一定是交换环（如：四元数环），交换的除环是体
	-	限制 $R$ 为幺环：保证 1 元满足作为 0 元单位元

-	唯一分解环：每个非零非可逆元都能唯一分解的整环

###	环案例

-	集环：满足以下任意条件之一的非空集集合 $R$ 是环，以交为乘法、对称差为加法、空集为零元
	-	$R$ 为集合并、差运算封闭
	-	$R$ 对集合交、对称差运算封闭
	-	$R$ 对集合交、差、无交并运算封闭

-	整数环、有理数环、实数域、复数域：交换、含单位环

-	多项式环：所有项系数构成环的多项式全体

-	方阵环

###	环的基本性质

-	$\forall a \in R, a \cdot 0 = 0 \cdot a = 0$
	-	$a \cdot 0 = a \cdot (0+0) = a \cdot 0 + a \cdot 0$
	-	$a \cdot 0 - a \cdot 0 = a \cdot 0 + a \cdot 0 - a \cdot 0$
	-	考虑到环有加法逆元，则有 $0 = a \cdot 0$

-	$\forall a,b \in R, (-a) \cdot b = a \cdot (-b) = -(a \cdot b)$
	-	$(-a) \cdot b + a \cdot b = (-a + a) \cdot b = 0$

##	环的理想

-	考虑环 $(R, +, \cdot)$ 上交换群 $(R, +)$，集合 $I \subseteq R$
	-	$(I, +)$ 构成 $(R, +)$ 的子群
	-	$\forall i \in I, r \in R$ 有 $i \cdot r \in I$
	-	$\forall i \in I, r \in R$ 有 $r \cdot i \in I$

-	环理想定义
	-	**右理想**：若 $I$ 满足上述 1、2，则称 $I$ 是 $R$ 的右理想
	-	**左理想**：若 $I$ 满足上述 1、3，则称 $I$ 是 $R$ 的左理想
	-	**（双边）理想**：若 $I$ 同时是 $R$ 的左理想、右理想
		-	交换环的理想都是双边理想
	-	**真左、右、双边理想**：$I \subset R$
	-	**极大左、右、双边理想**：不存在 $J \subset I$ 使得 $J$ 是 $R$ 的真左、右、双边理想
		-	极大双边理想不定是极大左、右理想

-	定理
	-	交换幺环 $R$ 中，理想 $I$ 是 $R$ 的极大理想的充要条件是：商环 $R \setminus I$ 是域
	-	$I$ 是环 $R$ 的左理想，则 $I$ 是 $R$ 的极大左理想的充要条件是对 $R$ 的任意不含在 $I$ 中的左理想的 $J$ 都有 $I+J=R$

> - 整数环 $Z$ 只有形如 $nZ$ 的理想

###	生成理想、主理想

-	环 $R$，子集 $A \subseteq R$，定义 $<A> = RA + AR + RAR +ZA$，则有

	> - $ZA = \{\sum_{k=1}^K m_k a_k: m_k \in Z, a_i \in A, n \geq 1\}$：$Z$ 为整数集
	> - $AR = \{\sum_{k=1}^K a_k r_k: r_k \in R, a_i \in A, n \geq 1\}$
	> - $RA = \{\sum_{k=1}^K r_k a_k: r_k \in R, a_i \in A, n \geq 1\}$
	> - $RAR = \{\sum_{k=1}^K r_k a_k r_k^{'}: r_k,r_k^{'} \in R, a_i \in A, n \geq 1\}$：$Z$ 为整数集

	-	$<A>$ 是环 $R$ 的理想，称为 $R$ 中由子集 $A$ **生成的理想**，$A$ 称为 $<A>$ 的生成元集
		-	$<A>$ 是 $R$ 中所有包含子集 $A$ 的理想的交，即包含 $A$ 的最小理想
		-	同一理想的生成元集可能不唯一
	-	生成理想特殊情况
		-	$R$ 为交换环时，$<A> = RA + ZA$
		-	$R$ 为幺环时，$<A> = RAR$
		-	$R$ 为交换幺环时，$<A> = RA$

####	主理想

-	主理想：由环中单个元素生成的理想
	-	整数环 $Z$ 中，由 $p$ 生成的主理想是极大理想的充要条件是 $p$ 是素数

-	素理想：真理想 $I$ 被称为环 $R$ 的素理想，若对于所有理想 $A, B \subseteq R$，有 $AB \subseteq I \Rightarrow A \subseteq I 或 B \subseteq I$

-	素环：若环 $R$ 的零理想时素理想，则称 $R$ 使素环（质环）
	-	无零因子环是素环
	-	交换环 $R$ 中，真理想 $I$ 是素理想的充要条件是 $R / I$ 是素环

-	半素理想：真理想 $I$ 被称为环 $R$ 的素理想，若对于所有理想 $A^2 \subseteq I \rightarrow A \subseteq I$

###	环理想性质

-	环中，左、右、双边理想的和与交依然是左、右、双边理想
-	除环中，左、右理想只有平凡左、平凡右理想
-	对于环 $R$ 的两个理想 $A$、$B$，记 $AB = \{\sum_k a_k b_k | a_k \in A, b_k \in B \}$，则有
	-	若 $A$ 是 $R$ 的左理想，则 $AB$ 是 $R$ 的左理想
	-	若 $B$ 是 $R$ 的右理想，则 $AB$ 是 $R$ 的右理想
	-	若 $A$ 是 $R$ 的左理想、$B$ 是 $R$ 的右理想，则 $AB$ 是 $R$ 的双边理想

###	环理想衍生

-	主理想环：每个理想都是主理想的整环

-	单环：极大理想是零理想的幺环

##	整环

-	整数加法群的子群 $nZ$ 的 $n$ 个左陪集（同余类）记为 $\bar r_n = \{mn+r|m \in Z\}, r=0,1,2,\cdots,n-1$
	-	完全剩余系 $R_n=\{r_0,r_1,\cdots,r_{n-1}\}$：从 $\bar 0_n, \bar 1_n, \cdots, \overline {n-1}_n$ 中分别任选元素构成集合
		-	最小非负完全剩余系 $R_n=\{0,1,2,\cdots,n-1\}$ （非负最小剩余即正余数）
		-	缩（简化）剩余系 $\Phi_n=\{c_0,c_1,\cdots,c_{\phi(n)}\}$：完全剩余系中与 $n$ 互质的数构成集合
		-	最小正缩余系：最小非负完全剩余系中与 $n$ 互质的数构成集合
	-	所有与 $n$ 互质的同余类构成一个群（反证法易证，参见欧拉定理）
		-	最小正缩余系即群中各同余类中选择最小整数构成集合

###	裴蜀定理

-	裴蜀定理：$\forall a,b \in Z$，$d$ 为 $a,b$ 最大公约数，则关于 $x,y$ 的线性丢番图方程 $ax+by=m$ 有整数解当且仅当 $d|m$
	-	方程有解时必然有无穷多个解
	-	特别的，$m=1$ 时仅当 $a,b$ 互质有解，此即为 *大衍求一术* 情况

-	证明方向
	-	若 $a,b$ 中有一个为 0，显然成立
	-	设 $A=\{xa + yb | x \in Z, y \in Z\}$
		-	显然 $A$ 不空：$|a| \in A \cap N^{+}$
	-	考虑到自然数集合良序， $A$ 中存在最小正元素 $d_0 = x_0a + y_0b$
		-	$\forall p \in A, p > 0, p=x_1a + y_1b$ 以带余除法形式记作 $p = qd_0 + r$
		-	则有 $r = p - qd_0 = (x_1 - qx_0)a + (y_1 - qy_0)b \in A$，则 $r=0$
		-	即 $d_0|p, \forall p \in A, p > 0$，特别的 $d_0|a, d_0|b$
	-	对 $a,b$ 的任意正公约数 $d$，设 $a=kd,b=ld$
		-	则有 $d_0=x_0a + y_0b=(x_0k + y_0l)d$，即 $d|d_0$，$d_0$ 为 $a,b$ 最大公约数
	-	记 $m=m_0d_0$，则方程通解为 $\{(m_0x_0 + \frac {kb} d, m_0y_0 - \frac {ka} d) | k \in Z\}$
		-	$(x_0,y_0)$ 为一组特解

> - 最大公约数可理解为变动幅度单位，则仅在线性丢番图方程右端为整数倍单位时有整数解
> - 裴蜀定理可以推广至任意环上，即当且仅当 $m$ 属于 $d$ 生成的主理想时，在环内有解
> - <https://zh.wikipedia.org/wiki/%E8%B2%9D%E7%A5%96%E7%AD%89%E5%BC%8F>

###	中国剩余定理

$$(S): \left \{ \begin{matrix}
x \equiv a_1, & (\mod m_1) \\
x \equiv a_2, & (\mod m_2) \\
\vdots \\
x \equiv a_n, & (\mod m_n)
\end{matrix} \right.$$

-	中国剩余定理：对以下一元线性同余方程组，整数的 $m_1,m_2,\cdots,m_n$ 两两互质，则对任意整数 $a_1,a_2,\cdots,a_n$ 有解
	-	解在 $M$ 取模的意义下唯一
	-	通解可写为 $x = kM + \sum_{i=1}^n a_i t_i M_i, k \in Z$，其中
		-	$M = \prod_{i=1}^n m_i$，$M_i = M/m_i, i=1,2,\cdots,n$
		-	$t_iM_i \equiv 1 (\mod m_i), i=1,2,\cdots,n$

-	证明逻辑
	-	解存在性、唯一性
		-	考虑同阶集合 $A={0,1,\cdots,M-1}$、$B={(b_1,b_2,\cdots,b_n) | 0 \leq b_i \leq m_i, i=1,2,\cdots,n}$
		-	考虑映射 $f: A \rightarrow B = (x \mod m_1, x \mod m_2, \cdots, x \mod m_n)$ 为单射
			-	若 $f(x) = f(x^{'})$，则 $\forall i,m_i | (x-x^{'})$，则 $M | (x-x^{'})$
			-	又 $-M < x - x^{'} < M$，则必有 $x - x^{'} = 0$
		-	则映射 $f$ 为双射，逆映射即为线性同余方程组的唯一解
	-	通解证明
		-	$\sum_{i=1}^n a_i t_i M_i$ 带入可验证为方程组的一个特解
		-	考虑到 $m_1,m_2,\cdots,m_n$ 两两互质，则任意两个解间相差 $M$ 整数倍
		-	即找到一个特解，根据 $m_1,m_2,\cdots,m_n$ 互质确定解的规律

####	大衍求一术

-	大衍求一术：求解 $t_i$ 的算法
	-	$t_i$ 存在性可类似中国剩余定理建立双射 $f(x)=xM_i \mod m_i, x=0,1,\cdots,m_i$ 证明
	-	大衍求一术类似辗转相除法思想，即最大公约数为 1 时辗转相除

> - 模逆元求解即为求解线性丢番图方程，更一般情况下解的存在性、通解由裴蜀定理给出
> - <https://zhuanlan.zhihu.com/p/272302805>

###	欧拉函数

-	欧拉（总计）函数 $\phi(n)$：小于等于 $n$ 的正整数数中，与 $n$ 互质的数的数目
	-	欧拉函数取值：$\phi(n) = p_1^{k_1 - 1} p_2^{k_2 - 1} \cdots p_r^{k_r - 1} (p_1 - 1)(p_2 - 1)\cdots(p_r - 1)$
		-	可以变换为 $\phi(n) = n(1 - \frac 1 {p_1}) \cdots (1 - \frac 1 {p_r})$
		-	$p_i$ 为互异的质因子，$k_i$ 为质因子的次数
		-	显然 $r=1$ 时，上述公式成立
	-	欧拉函数即 $n$ 的同余类构成的乘法群的阶

-	欧拉函数积性：若 $m,n$ 互质，则 $\phi(m,n) = \phi(m) \phi(n)$
	-	考虑 $N < mn, N = k_1 m + p = k_2 n + q$，则 $N$ 与 $m,n$ 的互质取决于 $(p,q)$
	-	考虑如下线性同余方程组，根据中国剩余定理可建立与 $N$、与 $(p,q)$ 的双射

		$$\left \{ \begin{matrix}
		N \equiv p, & (\mod m) \\
		N \equiv q, & (\mod n)
		\end{matrix} \right.$$

> - 欧拉函数的积性可用初等代数证明，即将 $0,1,\cdots,mn-1$ 排成 $m * n$ 矩阵后直接考虑 $\phi(mn)$
> - <https://zhuanlan.zhihu.com/p/35060143>

####	欧拉定理

-	欧拉定理：若 $a,n$ 互质，则有 $a^{\phi(n)} \equiv 1 (\mod n)$
	-	特别的，$n$ 为质数时，即为费马小定理 $\forall a, a^{n-1} \equiv 1 (\mod n)$

-	证明框架
	-	考虑子群 $nZ$ 的最小正缩系 $\Phi_n=\{c_0,c_1,\cdots,c_{\phi(n)}\}$
	-	若 $a,n$ 互质，则 $a\Phi_n$ 也是一个缩剩余系
		-	若 $ac_i \equiv ac_j (\mod n), i \neq j$，即 $a(c_i - c_j) \equiv 0 (\mod n), i \neq j$
		-	考虑到 $a,n$ 互质，则有 $c_i \equiv c_j (\mod n), i \neq j$，矛盾
	-	则有 $\prod_{i=1}^{\phi(n)} c_i \equiv \prod_{i=1}^{\phi(n)} ac_i \equiv a^{\phi(n)} \prod_{i=1}^{\phi(n)} c_i (\mod n)$
	-	考虑到 $\prod_{i=1}^{\phi(n)},n$ 互质，则有 $a^{\phi(n)} \equiv 1 (\mod n)$

-	应用说明
	-	欧拉定理常用于化简求解同余
		-	$a,n$ 不互质时，可考虑将 $n$ 拆分为质因子，建立线性同余方程组求解
	-	$a,n$ 互质时，欧拉定理直接有 $a^m \equiv a^{m \mod n} (\mod n)$






