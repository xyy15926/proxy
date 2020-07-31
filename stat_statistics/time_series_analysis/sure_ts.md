---
title: 确定性时序分析
tags:
  - 统计
categories:
  - 统计
date: 2019-03-21 17:27:37
updated: 2019-02-17 11:57:08
toc: true
mathjax: true
comments: true
description: 确定性时序分析
---

###	目的

-	克服其他因素干扰，单纯测度某个确定性因素（季节、趋势、
	交易日）的序列的影响
	-	因素分解方法

-	根据序列呈现的确定性特征，选择适当的方法对序列进行综合
	预测
	-	指数平滑预测方法

##	Time Series Decomposition

###	因素分解模型

因素分解方法认为，所有序列波动可以归纳为受到以下四种因素影响
（全部或部分），导致序列呈现不同的波动特征，即任何时间序列
可以用这4个因素的某个函数进行拟合$x_t = f(T_t, C_t, S_t, I_t)$

-	trend：序列呈现的长期递增、递减的变化趋势

-	circle：序列呈现的从高到低、在由低到高的反复循环波动
	-	很多经济、社会现象确实有循环周期，但是这个周期往往
		很长、长度不固定
	-	如何观测值序列不够长，没有包含多个周期，周期的一部分
		会和趋势重合，无法准确、完整地提取周期影响
	-	在经济学领域常用的周期有
		-	基钦周期：平均40个月
		-	朱格拉周期：平均10年
		-	库兹涅茨周期：平均20年
		-	康德拉季耶夫周期：平均53.3年

-	season：和季节变化相关的稳定周期波动

-	immediate：其他不能用确定性因素解释的序列波动

###	常用模型（函数）

-	加法模型：$x_t = T_t + C_t + S_t + I_t$
-	乘法模型：$x_t = T_t * C_t * S_t * I_t$
-	伪加法模型：$x_t = T_t * (S_t + D_t + I_s)$
-	对数加法模型：$log_{x_t} = log_{T_t} + log_{S_t} + log_{D_t} + log_{I_t}$

###	因素分解的问题、改进

有些社会、经济现象显示某些特殊日期是很显著的影响因素，但是
在传统因素分解模型中，没有被纳入研究

-	股票交易受交易日影响
-	超市销售受周末、节假日影响
-	交通、运输、旅游同样受到周末、节假日影响

因此，如果观察时期步足够长，人们将circle改为day，模型中周期
因素$C_t$替换为$D_t$即可

##	指数平滑预测

根据序列是否具有长期趋势、季节效应，可以把序列分为3大类

-	既没有长期趋势、又没有季节效应
-	只有长期趋势、没有季节效应
-	有季节效应，无论是否有长期趋势

###	简单指数平滑

####	简单移动平均

对无趋势、季节的水平平稳序列

-	可以认为序列在比较短时间内，序列取值比较稳定，序列值差异
	主要是随机波动造成
-	根据此假定，可以使用最近一段时间内平均值作为未来几期
	预测值

$$
\beign{align*}
\hat x_{t+1} & = \frac {x_t + x_{t-1} + \dots + x_{t_n+1}} n \\
\hat x_{t+2} & = \frac {\hat x_t + x_t + x_{t-1} + \dots + x_{t-n+2} n \\
\hat x_{t+l} & = \frac {\hat x_{t+1-1} + \hat x_{t+l-2} + \dots +
	\hat x_{t+1} + x_t + \dots + x_{t-n+l} n \\
\end{align*}
$$

####	简单指数平滑预测

-	简单移动平均假定无论时间远近，n期的序列观测值影响力一样
-	但是在实际生活中，一般的随机事件而言，都是近期的结果对
	现在的影响大一些，远期结果对现在影响较小
-	指数平滑法构造思想：考虑到事件间隔对事件发展的影响，各期
	权重随时间间隔增大而指数衰减

$$
\begin{align*}
\hat x_{t+1} & = \alphax_t + \alpha(1-\alpha)x_{t-1} +
	\alpha(1-\alpha)^2 x_{t-2} + \dots
& = \alphax_t (1-\alpha)[\alphax_{t-1} + \alpha(1-\alpha)x_{t-2} +
	\alpha(1-\alpha)^2 x_{t-3} + \dost]
& = \alphax_t + (1-\alpha)\hat x_t
\end{align*}
$$

-	初值：很多方法可以确定，最简单指定$\hat x_1 = x_1$
-	平滑系数$\alpha$
	-	对于变化较缓慢的序列，取较小值
	-	对于变化迅速的序列，去较大值
	-	经验值在0.05 ~ 0.3，一般如果$\alpha$过大，说明序列
		波动性过强，不适合使用简单指数平滑
-	理论上可以预测任意期值，但是任意期预测值都是常数
	-	因为没有新的观测值提供新信息

###	Holt两参数指数平滑

适合对含有线性趋势的序列进行修匀，即分别用指数平滑的方法，
结合序列最新观察值，不断修匀参数a、b的估计值

$$
\begin{align*}
x_t & = a_0 + bt + \epsilon_t \\
& = a_0 + b(t-1) + b + \epsilon \\
& = (x_{t-1} + \epsilon_{t-1}) + b + \epsilon_t \\
& = a(t-1) + b(t)
\end{align*}
其中：a(t-1) = x_{t-1} - \epsilon_{t-1} \\
b(t) = b + \epsilon_t
$$

####	两参数递推公式

$$
\beign{align*}
\hat a(t) & = \alpha x_t + (1-\alpha)[\hat \alpha(t-1) + \hat b(t-1)] \\
\hat b(t) & = \beta[\hat a(t) - \hat a(t-1)] + (1-\beta)\hat b(t-1)
\end{align*}
$$

####	序列预测公式

$$
\hat x_{t+k} = \hat a(t) + \hat b(t)*k \\
初值：\hat a(0)=x_1, \hat b(0)=\frac {x_{n+1} - x_1} n
$$


###	Holt-Winter三参数指数平滑

在Holt指数平滑的基础上构造了Holt-Winters三参数指数平滑，以
修匀季节效应

####	加法模型

假定：
$$
\begin{aling*}
x_t & = a_0 + bt + c_t + \epsilon_t \\
& = a_0 + b(t-1) + b + c_t + \epsilon_t \\
& = (x_{t-1} - c{t-1} - \epsilon_{t-1}) + b +
	\epsilon_t + (Sd_j + e_t) \\
& = a(t-1) + b(t) + c(t)

其中：a(t-1) = x_{t-1} - c_{t-1} - \epsilon_{t-1} \\
b(t) = b + \epsilon_t \\
c_t = Sd_t + e_t, e_t ~ N(0, \sigma_e^2) \\
\end{align*}
$$

递推式、预测：

$$
\begin{align*}
\hat a(t) & = \alpha(x_t - c(t-s)) +
	(1-\alpha)[\hat a(t-1) + \hat b(t-1)] \\
\hat b(t) & = \beta[\hat a(t) - \hat a(t-1)] +
	(1-\beta)\hat b(t-1) \\
\hat c(t) = \gamma[x_t - \hat a(t)] + (1-\gamma)c(t-s) \\

预测：\hat x_{t+k} & = \hat a(t) + \hat b(t) +
	\hat c(t + mod(k,s) -s) \\
\end{align*}
$$


####	乘法模型

假定：
$$
\begin{aling*}
x_t & = (a_0 + bt + \epsilon_t)c_t \\
& = (a_0 + b(t-1) + b + \epsilon_t)c_t \\
& = [(x_{t-1}/c{t-1} - \epsilon_{t-1})+
	(b + \epsilon_{t-1})](S_j + e_t) \\
& = [a(t-1) + b(t)]c(t)

其中：a(t-1) = x_{t-1}/c_{t-1} - \epsilon_{t-1} \\
b(t) = b + \epsilon_t \\
c_t = S_j + e_t, e_t ~ N(0, \sigma_e^2) \\
\end{align*}
$$

递推式、预测：

$$
\begin{align*}
\hat a(t) & = \alpha(x_t / c(t-s)) +
	(1-\alpha)[\hat a(t-1) + \hat b(t-1)] \\
\hat b(t) & = \beta[\hat a(t) - \hat a(t-1)] +
	(1-\beta)\hat b(t-1) \\
\hat c(t) = \gamma[x_t / \hat a(t)] + (1-\gamma)c(t-s) \\

预测：\hat x_{t+k} & = [\hat a(t) + \hat b(t) * k]
	\hat c(t + mod(k,s) -s) \\
\end{align*}
$$

