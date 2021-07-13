---
title: 统计检验
categories:
  - Math Mixin
  - Time Series
tags:
  - Statistics
  - Time Series
  - Statistic Tests
date: 2019-03-21 17:27:37
updated: 2019-02-17 11:57:08
toc: true
mathjax: true
comments: true
description: 统计检验
---

##	*JJ* 检验

###	检验思想

-	*JJ* 检验：检验 $VAR(k)$ 模型的协整关系
	-	参考多元分析中典型相关分析的构造思想

$$\begin{align*}
Y_t & = c + \Pi_1 Y_{t-1} + \Pi_2 Y_{t-2} + \cdots +
	\Pi_k Y_{t-k} + u_t + \Phi D_t \\
\Delta Y_t & = y_{t-1} + \Pi Y_{t-1} + \Gamma_1 \Delta Y_{t-2}
	+ \cdots + \Gamma_{k-1} \Delta Y_{t-p+1} + \epsilon_t + \Phi D_t
\end{align*}$$

> - $Y_t = (y_{1,t}, y_{2,t}, \cdots, y_{N,t})^T \sim I(1)$
> - $\Pi = \sum_{i=1}^k \Pi_i - I$
> - $\Gamma_i = -\sum_{j=i+1}^k$

-	基础协整关系 = $\Pi$ 非零特征根数量
	-	基础协整关系的任意线性组合依然是协整的
	-	系统中协整关系至少为非零特征根数量

###	检验方法

>  - 假设 $\lambda_1 \geq \lambda_2  \geq \cdots \lambda_m$ 是 $\Pi$ 的所有特征根

####	最大特征根检验


-	检验统计量：*Bartlette* 统计量 $Q = -Tln(1-\lambda_i^2)$

-	假设
	-	原假设：$H_0: \lambda_i = 0$

-	检验流程
	-	从 $\lambda_1$ 开始检验是否显著不为 0
	-	直到某个 $\lambda_k$ 非显著不为0，则系统有 $k-1$ 个协整关系

####	迹检验

-	检验统计量：*Bartlette* 统计量 $Q = -T \sum_{j=i}^m ln(1-\lambda_j^2)$

-	假设
	-	原假设：$H_0: \sum_{j=i}^m \lambda_j = 0$

-	检验流程
	-	从 $\sum_{j=1}^m \lambda_j = 0$ 开始检验是否显著不为 0
	-	直到某个 $\sum_{j=k}^m ln(1-\lambda_j^2)$ 非显著不为 0，说明系统存在$k-1$个协整关系

