---
title: 统计量 - 相关
categories:
  - Math Mixin
  - Statistics
tags:
  - Statistics
  -	Correlation
date: 2021-07-12 09:53:01
updated: 2021-07-12 09:53:01
toc: true
mathjax: true
description: 
---

##	*Pearson* 积矩相关系数

$$
\rho_{X,Y} = \frac {cov(X, Y)} {\sigma_X \sigma_Y}
$$

> - $cov(X, Y)$：变量 $X, Y$ 协方差
> - $\sigma_X, \sigma_Y$：变量 $X, Y$ 方差

-	*Pearson* 积矩相关系数取值范围为 $[-1, 1]$
	-	$1, -1$ 分别表示变量成正线性、负线性函数关系

###	显著性检验

####	*Fisher* 变换

$$
z = \frac 1 2 ln(\frac {1+r} {1-r}) = arctanh(r)
$$

> - $z$：*Pearson* 积矩相关系数的 *Fisher* 变换
> - $r$：样本的 *Pearson* 积矩相关系数值

-	当 $(X, Y)$ 为二元正态分布时，$z$ 近似正态分布
	-	均值：$\frac 1 2 ln(\frac {1+\rho} {1-\rho})$
	-	标准差：$\frac 1 {\sqrt {N - 3}}$

####	基于数学的近似方法

$$
t = r \sqrt{\frac {N - 2} {1 - r^2}}
$$

-	当 $(X, Y)$ 为二元正态分布且不相关时，$t$ 服从自由度为 $n-2$的 *t-分布*

##	*Spearman* 秩相关系数

$$\begin{align*}
\rho_{X, Y} & = \frac {cov(Rank(X) - Rank(Y))}
	{\sigma_{Rank(X)} \sigma_{Rank(Y)}} \\
& = 1 - \frac {6 \sum_i^N d_i^2} {N(N^2-1)} \\
\end{align*}$$

> - $Rank(X), Rank(Y)$：变量 $X, Y$ 的秩（应同序）（相同值秩取均值）
> - $d_i$：变量对 $X, Y$ 中，二者秩差值

-	*Spearman* 秩相关系数被定义为变量秩的 *Pearson* 相关系数

> - *Spearman* 秩相关系数也可以使用 *Fisher* 变换检验显著性

##	*Kendell* 秩相关系数

$$\begin{align*}
\tau_a &= \frac {N_c - N_d} {N_0} \\
\tau_b &= \frac {N_c - N_d} {\sqrt{(N_0 - N_X)(N_0 - N_Y)}} \\
\tau_c &= \frac {2(N_c - N_d)} {N^2 \frac {M-1} M}
\end{align*}$$

> - $N_0 = \frac {N(N-1)} 2$：变量对数量
> - $N_c, N_d$：变量对 $X, Y$ 中有序对数量、无序对数量
> - $N_X, N_Y$：变量对 $X, Y$ 中 $X$ 取值、$Y$ 取值相同对数量
> - $M$：变量 $X, Y$ 中较小取值数量者取值数量

-	*Kendell* 秩相关系数取值范围同样为 $[-1, 1]$
	-	-1 仅在变量 $X, Y$ 取值完全反向取到

-	$\tau_a$ 是 $\tau_b$ 在变量不存在取值相同时的特例

-	$\tau_c$ 适合“层级”数据，即两个变量取值类似划分、内部细分

	||A|B|C|
	|-----|-----|-----|-----|
	|I-1|30|0|0|
	|I-2|30|0|0|
	|II-1|0|30|0|
	|II-1|0|30|0|
	|III-2|0|0|30|
	|III-2|0|0|30|

	-	对以上数据，$\tau_b$ 取值在 0.9 附近，而 $\tau_c$ 取 1

> - 有序对：对 $(X_i, Y_i), (X_j, Y_j)$，满足 $X_i < X_j, Y_i < Y_j$ 或 $X_i > X_j,Y_i > Y_j$ 则为有序对
> - 无序对：对$(X_i, Y_i), (X_j, Y_j)$，满足 $X_i < X_j, Y_i > Y_j$ 或 $X_i > X_j, Y_i < Y_j$ 则为无序对

##	卡方统计量

卡方统计量：通过观察实际与理论值的偏差确定理论正确与否

$$
\chi^2 = \sum \frac {(A - E)^2} E
$$

> - $A$：自变量、因变量组合对应频数观察值
> - $E$：自变量、因变量组合对应频数期望值

-	将模型预测结果视为实际分布、先验分布（均匀分布）视为理论分布

-	卡方检验：检验定性变量之间相关性，假设两个变量确实独立，观察实际值、理论值偏差程度判断变量之间相关性
	-	若偏差足够小，认为误差是自然的样本误差，两者确实独立
	-	若偏差大到一定程度，误差不可能由偶然、测量精度导致，
		认为两者相关

-	若模型预测结果同先验分布差别很大，说明模型有效，且卡方统计量值越大表示预测把握越大

###	特点

-	由于随机误差存在，卡方统计量容易
	-	夸大频数较小的特征影响
	-	相应的，取值数较少（各取值频数相对而言可能较大）特征影响容易被低估

###	分布证明

-	考虑随机变量 $X=(x_1,\cdots,x_D)$ 服从 *Multinomial* 分布，分布参数为 $n, p=(p_1,\cdots,p_D)$

-	考虑服从理论分布的随机变量 $X$ 协方差矩阵

	$$\begin{align*}
	\Sigma = Cov(X) &= \begin{bmatrix}
		np_1(1-p_1) & -np_1p_2 & \cdots & -np_1p_D \\
		np_2p_1 & -np_2(1-p_2) & \cdots & -np_2p_D \\
		\vdots & \vdots & \ddots & \vdots \\
		-np_Dp_1 & -np_Dp_2 & \cdots & np_D(1-p_D)
	\end{bmatrix} \\
	&= n\begin{bmatrix}
		p_1 & 0 & \cdots & 0 \\
		0 & p_2 & \cdots & 0 \\
		\vdots & \vdots & \ddots & \vdots \\
		0 & 0 & \cdots & p_D
	\end{bmatrix} - npp^T \\
	\end{align*}$$

-	则由中心极限定理有，如下依分布收敛的结论

	$$\begin{align*}
	\frac {(X - np)} {\sqrt n} & \overset {D} {\rightarrow} N(0,\Sigma) \\
	\end{align*}$$

-	考虑服从理论分布的随机变量 $X$ 的 $\chi^2$ 参数

	$$\begin{align*}
	\chi^2 &= \frac 1 n (X-np)^T D^2 (X-np) \\
	D &= \begin{bmatrix}
		\frac 1 {\sqrt {p_1}} & 0 & \cdots & 0 \\
		0 & \frac 1 {\sqrt {p_2}} & \cdots & 0 \\
		\vdots & \vdots & \ddots & \vdots \\
		0 & 0 & \cdots & \frac 1 {\sqrt {p_D}}
	\end{bmatrix}
	\end{align*}$$

-	并由连续映射定理可以得到 $D\frac {x-np} {\sqrt n}$ 分布，且其协方差矩阵 $\Sigma_0$ 满足

	$$\begin{align*}
	D\frac {x-np} {\sqrt n} & \overset {D} {\rightarrow} N(0, D \Sigma D^T) \\
	\Sigma_0 &= D \Sigma D^T \\
	\Sigma_0^2 &= (E - \sqrt p {\sqrt p}^T)(E - \sqrt p {\sqrt p}^T) = \Sigma_0 \\
	\end{align*}$$

-	由以上，$\Sigma_0$ 仅有特征值 0，1
	-	特征值 0 对应特征向量有且仅有 $\sqrt p$
	-	特征值 1 对应特征向量有 $D-1$ 个

	$$\begin{align*}
	\Sigma_0 \sqrt p - 0 \sqrt p &= \sqrt p - \sqrt p = 0 \\
	\Sigma_0 \lambda - 1 \lambda &= \lambda - \sqrt p {\sqrt p}^T \lambda
		= \sqrt p {\sqrt p}^T \lambda = 0
	\end{align*}$$

-	则 $\chi^2$ 统计量依分布收敛于自由度为 $D-1$ 的卡方分布

	$$\begin{align*}
	\chi^2 &= \sum_{d=1}^D \frac {(x_d - np_d)^2} {np_d}
		\overset {D} {\rightarrow} \chi_{D-1}
	\end{align*}$$

-	可据此构造统计量进行卡方检验，检验实际值实际分布频率 $(a_1,\cdots,a_D)$ 是否符合该分布
	-	构造卡方统计量 $\chi^2 = \sum_{d=1}^D \frac {(x_d - na_d)^2} {na_d}$
	-	则卡方统计量在随机变量满足多项分布情况下依分布收敛于自由度为 $D-1$ 的卡方分布

> - <https://www.zhihu.com/question/309694332/answer/952401910>
> - <https://zhuanlan.zhihu.com/p/198864907>

