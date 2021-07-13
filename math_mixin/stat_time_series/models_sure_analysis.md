---
title: 确定性时序分析
categories:
  - Math Mixin
  - Time Series
tags:
  - Statistics
  - Time Series
  - Time Series Decomposition
  - Exponential Smoothing
date: 2019-03-21 17:27:37
updated: 2021-07-12 20:44:40
toc: true
mathjax: true
comments: true
description: 确定性时序分析
---

##	*Time Series Decomposition*

> - 因素分解方法：克服其他因素干扰，单纯测度某个确定性因素（季节、趋势、交易日）的序列的影响
> - 指数平滑预测方法：根据序列呈现的确定性特征，选择适当的方法对序列进行综合预测


###	因素分解模型

-	因素分解模型思想
	-	所有序列波动可以归纳为受到以下 4 种因素影响（全部或部分）
	-	导致序列呈现不同的波动特征，即任何时间序列可以用 4 因素的某个函数进行拟合 $x_t = f(T_t, C_t, S_t, I_t)$

-	*Trend*：序列呈现的长期递增、递减的变化趋势

-	*Circle*：序列呈现的从高到低、在由低到高的反复循环波动
	-	很多经济、社会现象确实有循环周期，但是这个周期往往很长、长度不固定
	-	如何观测值序列不够长，没有包含多个周期，周期的一部分会和趋势重合，无法准确、完整地提取周期影响
	-	在经济学领域常用的周期有
		-	基钦周期：平均 40 个月
		-	朱格拉周期：平均 10 年
		-	库兹涅茨周期：平均 20 年
		-	康德拉季耶夫周期：平均 53.3 年

-	*Season*：和季节变化相关的稳定周期波动

-	*Immediate*：其他不能用确定性因素解释的序列波动

###	常用模型（函数）

-	加法模型：$x_t = T_t + C_t + S_t + I_t$
-	乘法模型：$x_t = T_t * C_t * S_t * I_t$
-	伪加法模型：$x_t = T_t * (S_t + D_t + I_s)$
-	对数加法模型：$log_{x_t} = log_{T_t} + log_{S_t} + log_{D_t} + log_{I_t}$

###	考虑节假日

-	有些社会、经济现象显示某些 **特殊日期** 是很显著的影响因素，但是在传统因素分解模型中，没有被纳入研究

	-	股票交易受交易日影响
	-	超市销售受周末、节假日影响
	-	交通、运输、旅游同样受到周末、节假日影响

-	如果观察时期不足够长，考虑将模型中 *Circle*（周期） 改为 *Day*（节假日）

##	*Exponential Smoothing*

-	根据序列是否具有长期趋势、季节效应，可以把序列分为3大类
	-	既没有长期趋势、又没有季节效应
	-	只有长期趋势、没有季节效应
	-	有季节效应，无论是否有长期趋势

###	简单指数平滑

####	简单移动平均

-	对无趋势、季节的水平平稳序列
	-	可以认为序列在比较短时间内，序列取值比较稳定，序列值差异主要是随机波动造成
	-	根据此假定，可以使用最近一段时间内平均值作为未来几期预测值

$$\begin{align*}
\hat x_{t+1} & = \frac {x_t + x_{t-1} + \dots + x_{t_n+1}} n \\
\hat x_{t+2} & = \frac {\hat x_t + x_t + x_{t-1} + \dots + x_{t-n+2}} n \\
\hat x_{t+l} & = \frac {\hat x_{t+1-1} + \hat x_{t+l-2} + \dots +
	\hat x_{t+1} + x_t + \dots + x_{t-n+l}} n \\
\end{align*}$$

> - 简单移动平均假定无论时间远近，近 $n$ 期的序列观测值影响力一样

####	简单指数平滑预测

> - 实务中，对一般的随机事件，近期的结果对现在的影响更大

-	指数平滑法构造思想
	-	考虑到事件间隔对事件发展的影响，各期权重随时间间隔增大而指数衰减

$$\begin{align*}
\hat x_{t+1} & = \alpha x_t + \alpha (1-\alpha) x_{t-1} +
	\alpha (1-\alpha)^2 x_{t-2} + \dots \\
& = \alpha x_t (1-\alpha) [\alpha x_{t-1} + \alpha (1-\alpha) x_{t-2} +
	\alpha (1-\alpha)^2 x_{t-3} + \dots] \\
& = \alpha x_t + (1-\alpha) \hat x_t
\end{align*}$$

-	初值：很多方法可以确定，最简单指定 $\hat x_1 = x_1$
-	平滑系数 $\alpha$
	-	经验值在 $[0.05, 0.3]$，
		-	对于变化较缓慢的序列，取较小值
		-	对于变化迅速的序列，取较大值
	-	如果 $\alpha$ 过大，说明序列波动性过强，不适合使用简单指数平滑
-	理论上可以预测任意期值，但是任意期预测值都是常数
	-	因为没有新的观测值提供新信息

###	*Holt* 两参数指数平滑

-	两参数指数平滑
	-	适合对含有线性趋势的序列进行修匀
	-	即分别用指数平滑的方法，结合序列最新观察值，不断修匀参数 $a, b$ 的估计值

$$\begin{align*}
x_t & = a_0 + bt + \epsilon_t \\
& = a_0 + b(t-1) + b + \epsilon \\
& = (x_{t-1} + \epsilon_{t-1}) + b + \epsilon_t \\
& = a(t-1) + b(t)
\end{align*}$$

> - $a(t-1) = x_{t-1} - \epsilon_{t-1}$
> - $b(t) = b + \epsilon_t$

-	两参数递推公式

	$$\begin{align*}
	\hat a(t) & = \alpha x_t + (1-\alpha)[\hat \alpha(t-1) + \hat b(t-1)] \\
	\hat b(t) & = \beta [\hat a(t) - \hat a(t-1)] + (1-\beta) \hat b(t-1)
	\end{align*}$$

-	序列预测公式

	$$
	\hat x_{t+k} = \hat a(t) + \hat b(t)*k \\
	$$

-	初值设置
	-	$\hat a(0)=x_1$
	-	$\hat b(0)=\frac {x_{n+1} - x_1} n$

###	*Holt-Winter* 三参数指数平滑

-	三参数指数平滑
	-	在 *Holt* 指数平滑的基础上构造，以修匀季节效应

####	加法模型

-	模型表达式

	$$\begin{align*}
	x_t & = a_0 + bt + c_t + \epsilon_t \\
	& = a_0 + b(t-1) + b + c_t + \epsilon_t \\
	& = (x_{t-1} - c{t-1} - \epsilon_{t-1}) + b + \epsilon_t + (Sd_j + e_t) \\
	& = a(t-1) + b(t) + c(t)
	\end{align*}$$

	> - $a(t-1) = x_{t-1} - c_{t-1} - \epsilon_{t-1}$
	> - $b(t) = b + \epsilon_t$
	> - $c_t = Sd_t + e_t, e_t \sim N(0, \sigma_e^2)$

-	三参数递推式

	$$\begin{align*}
	\hat a(t) & = \alpha(x_t - c(t-s)) + (1-\alpha)[\hat a(t-1) + \hat b(t-1)] \\
	\hat b(t) & = \beta[\hat a(t) - \hat a(t-1)] + (1-\beta)\hat b(t-1) \\
	\hat c(t) & = \gamma[x_t - \hat a(t)] + (1-\gamma)c(t-s)
	\end{align*}$$

-	序列预测公式

	$$
	\hat x_{t+k} = \hat a(t) + \hat b(t) + \hat c(t + mod(k,s) -s)
	$$


####	乘法模型

-	模型表示式

	$$\begin{align*}
	x_t & = (a_0 + bt + \epsilon_t)c_t \\
	& = (a_0 + b(t-1) + b + \epsilon_t)c_t \\
	& = [(x_{t-1}/c{t-1} - \epsilon_{t-1})+
		(b + \epsilon_{t-1})](S_j + e_t) \\
	& = [a(t-1) + b(t)]c(t)
	\end{align*}$$

	> - $a(t-1) = x_{t-1}/c_{t-1} - \epsilon_{t-1}$
	> - $b(t) = b + \epsilon_t$
	> - $c_t = S_j + e_t, e_t \sim N(0, \sigma_e^2)$

-	三参数递推式

	$$\begin{align*}
	\hat a(t) & = \alpha(x_t / c(t-s)) + (1-\alpha)[\hat a(t-1) + \hat b(t-1)] \\
	\hat b(t) & = \beta[\hat a(t) - \hat a(t-1)] + (1-\beta)\hat b(t-1) \\
	\hat c(t) & = \gamma[x_t / \hat a(t)] + (1-\gamma)c(t-s)
	\end{align*}$$

-	序列预测公式

	$$
	\hat x_{t+k} = [\hat a(t) + \hat b(t) * k] \hat c(t + mod(k,s) -s)
	$$

