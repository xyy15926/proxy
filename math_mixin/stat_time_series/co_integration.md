---
title: 协整与误差修正模型
categories:
  - Math Mixin
  - Time Series
tags:
  - Statistics
  - Time Series
  - Regression
  - Spurious Regression
date: 2019-03-21 17:27:37
updated: 2019-02-17 11:57:08
toc: true
mathjax: true
comments: true
description: 协整与误差修正模型
---

##	*Spurious Regression*

-	多变量分析中，平稳性非常重要，忽略序列平稳性判断，容易出现伪回归现象

-	*Granger*、*Newbold* 的非平稳序列的伪回归随机模型实验（两个独立随机游走模型）表明
	-	非平稳场合，参数显著性检验犯弃真错误的概率远大于 $\alpha$，伪回归显著成立
	-	即 $P(|t| \geqslant t_{\alpha/2}(n) | 非平稳序列) \leqslant \alpha$

###	*Cointegration* 协整关系

$$
y_t = \beta_0 + \sum_{i=1}^k \beta_i x_{it} + \epsilon_t
$$

> - ${x_1}, {x_2}, \cdots, {x_k}$：自变量序列
> - $y_t$：响应变量序列
> - ${\epsilon_t}$：平稳回归残差序列

####	协整检验

-	假设条件
	-	$H_0: \epsilon_t ~ I(k), k \geqslant 1$：多元非平稳序列之间不存在协整关系
	-	$H_1: \epsilon_t ~ I(0)$：多元非平稳序列之间存在协整关系

-	建立响应序列与输入序列之间的回归模型
-	对回归残差序列进行 *EG* 平稳性检验

###	*Error Correction Model*

*ECM*：误差修正模型，解释序列短期波动关系

-	*Granger* 证明协整模型、误差修正模型具有 *1-1* 对应关系
	-	协整模型度量序列之间长期均衡关系

> - 实务中，响应序列与解释序列很少处于均衡点上，实际观测的是序列间短期或非均衡关系

####	*Granger* 表述定理

> - 如果变量 $X$、$Y$ 是协整的，则他们之间的短期非均衡关系总能用一个误差修正模型表述 $$
		\Delta Y_t = lagged(\Delta Y, \Delta X) - \lambda ECM_{t-1} + \epsilon_t
	$$

-	对关系 $y_t = \beta x_t + \epsilon_t$

	$$\begin{align*}
	y_t - y_{t-1} & = \beta x_t - \beta x_{t-1} - \epsilon_{t-1} + \epsilon_t \\
	\Delta y_t & = \beta \delta x_t - ECM_{t-1} + \epsilon_t
	\end{align*}$$

-	响应序列当期波动 $\Delta y_t$ 主要受到三方面短期波动影响
	-	$\Delta x_t$：输出序列当前波动
	-	$ECM_{t-1}$：上一期误差
	-	$\epsilon_t$：纯随机波动

####	误差修正模型

$$
\Delta y_t = \beta_0 \Delta x_t + \beta_1 ECM_{t-1} + \epsilon_t
$$

-	$\beta_1 < 0$：表示负反馈机制
	-	$ECM_{t-1} > 0$：正向误差，则会导致下一期值负向变化
	-	$ECM_{t-1} < 0$：负向误差，则会导致下一期值正向变化

###	*Granger* 因果关系

-	因果关系：原因导致结果
	-	时间角度：原因发生在前，结果发生在后
	-	影响效果：$X$ 事件发生在前，且对 $Y$ 事件发展结果有意义

-	*Granger* 检验可检验统计学意义上的 *Granger* 因果关系
	-	统计意义上的因果关系和现实意义上因果关系不同
	-	现实意义上变量因果关系强调逻辑自洽

####	*Granger* 因果关系

> - 序列 $X$ 是序列 $Y$ 的 *Granger* 原因，当且仅当最优线性预测函数使得下式成立 $$
		\theta^2(y_{t+1}|I_t) \leq \theta^2(y_{t+1}|I_t-X_t)
	$$

> - $I_t = \{ x_t, x_{t-1}, \cdots, y_t, y_{t-1}, \cdots \}$：$t$ 时刻所有有用信息集合
> - $X_t = \{ x_t, x_{t-1}, \cdots \}$：t时刻所有序列信息集合
> -	$\theta^2(y_{t+1}|I_t)$：使用所有可获得历史信息 （包括 ${x}$ 序列历史信息）得到的一期预测值方差
> -	$\theta^2(y_{t+1}|I_t-X_t)$：从所有信息中刻意扣除 ${x}$ 序列历史信息得到的一期预测值方差

-	*Granger* 因果关系分类
	-	$(x, y)$：相互独立
	-	$(x \leftarrow y)$：$x$ 是 $y$ 的 *Granger* 原因
	-	$(x \rightarrow y)$：$y$ 是 $x$ 的 *Granger* 原因
	-	$(x \leftrightarrow y)$：互为因果

####	*Granger* 因果检验

-	建立回归方程

	$$
	y_t = \alpha_0 + \sum_{i=1}^m \alpha_i x_{t-i} +
		\sum_{j=1}^n b_j y_{t-j} + \sum_k cz_{t-k} +
		\epsilon_t
	$$

	> - $z_t$：其他解释变量集合
	> - $\epsilon_t \sim I(0)$

-	假设
	-	$H_0: \alpha_1 = \alpha_2 = \cdots = \alpha_m = 0$
	-	$H_1: \alpha_i 不全为0$

-	检验统计量：F统计量

	$$
	F = \frac {(SSE_r - SSE_u) / q} {SSE_u/n-q-p-1} \sim F(q, n-p-q-1)
	$$

####	*Granger* 因果检验说明

-	*Granger* 因果检验思想：对响应变量预测精度有显著提高的自变量，就视为响应变量的因
	-	因果性可以推出预测精度提高，但预测精度提高不能等价推出因果性
	-	即使检验结果显著拒绝原假设，也不能说明两个序列之间有真正因果关系

-	*Granger* 因果检验是处理复杂变量关系时的工具
	-	借助因果检验信息，可以帮助思考模型结果
	-	不一定准确，但是提供信息比完全没有信息好

-	*Granger* 因果结果说明
	-	检验结果严重依赖解释变量的延迟阶数，不同延迟阶数可能会得到不同的检验结果
	-	检验结果会受到样本随机性影响，样本容量越小随机性越大，所以最好在样本容量比较大时进行检验





