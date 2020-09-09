---
title: 常用统计量
tags:
  - 机器学习
categories:
  - 机器学习
date: 2019-07-14 20:04:44
updated: 2019-07-14 20:04:44
toc: true
mathjax: true
comments: true
description: 常用统计量
---

##	统计量

统计量：统计理论中对数据进行分析、检验的变量

-	传统的统计量具有显式解析表达式
	-	均值：数据之和除数量
	-	中位数：数据中间者

-	统计量同样可以理解为和数据相关**优化问题的解**
	-	均值：离差平方和最小
	-	中位数：划分均匀

	> - 优化问题目标本身也是统计量

##	Gini指数

基尼指数

$$\begin{align*}
Gini(p) & = \sum_{k=1}^K p_k(1-p_k) \\
	& = 1 - \sum_{k=1}^K p_k^2
\end{align*}$$

> - $p$：概率分布
> - 异质性最小：Gini系数为0
> - 异质性最大：Gini系数为$1 - \frac 1 k$

-	Gini指数度量分布的不纯度
	-	包含类别越多，Gini指数越大
	-	分布越均匀，Gini指数越大

> - 熵较Gini指数对不纯度判罚更重

![gini_entropy_error_rate_in_binary_classification](imgs/gini_entropy_error_rate_in_binary_classification.png)

###	条件Gini指数

条件Gini指数

$$
Gini(Y|X) = \sum_{k=1}^K P(X=x_k)Gini(Y|X=x_k)
$$

> - 性质类似信息增益

##	Entropy

熵：则是在概率分布上对随机变量X的不确定性/混乱程度的度量

> - 信息熵：信息消除不确定性的度量

$$
\begin{align*}
H(X) & = -E_P log P(x) \\
& = \sum_i^N P(x_i) log \frac 1 {P(x_i)} \\
& = - \sum_i^N p_i log p_i \\
\end{align*}
$$


> - $u_i$：随机变量各个取值
> - $p_i$：随机变量各取值对应概率
> - 事件i发生概率$p_i=0$：约定$p_i log(p_i)$为0
> - 其中$log$以2为底，单位为*bit*，以e为底，单位为*nat*

-	熵只依赖随机变量$X$的分布，与其取值无关，所以也可以将其
	熵记为$H(P)$

-	由定义$0 \leq H(P) \leq log_2 k$
	-	$H(p) = 0$：$\exists j, p_j=1$，随机变量只能取
		一个值，无不确定性
	-	$H(p) = log k$：$\forall j, p_j=1/k$，随机变量
		在任意取值概率相等，不确定性最大

-	熵可以视为变量取值概率的加权和

> - *empirical entropy*：经验熵，熵中的概率由数据估计时
	（尤极大似然估计）

###	熵的性质

-	一般的
	$$\begin{align*}
	H(X, Y) & = H(X) + H(Y|X) \\
	H(X, Y) & = H(Y) + H(X|Y) \\
	H(X|Y) & \leqslant H(x) \\
	H(X, Y) & \leqslant H(X) + H(Y) \\
	\end{align*}$$

-	特别的，若X、Y相互独立
	$$
	H(X, Y) = H(X) + H(Y)
	$$

###	*Conditinal Entrophy*

条件熵：随机变量X给定条件下，随机变量Y的**条件概率分布的熵**
对X的数学期望

$$\begin{align*}
H(Y|X) & = \sum_{i=1}^N p_i H(Y|X=x_i) \\
H(Y|x=x_i) & = - \sum_j P(y_j|x_i) log P(y_j|x_i)
\end{align*}$$

> - $P(X=x_i, Y=y_j)=p_{i,j}$：随机变量$(X,Y)$联合概率分布
> - $p_i=P(X=x_i)$
> - $H(Y|X=x_i)$：后验熵

> - *postorior entropy*：后验熵，随机变量X给定条件下，随机
	变量Y的**条件概率分布的熵**
> - *empirical conditional entropy*：经验条件熵，概率由数据
	估计

###	*Mutual Infomation*/*Infomation Gain*

互信息/信息增益：（经验）熵与（经验）条件熵之差

$$\begin{align*}
g(Y|X) & = H(Y) - H(Y|X) \\
& = \sum_{x \in X} \sum_{y \in Y} p(x,y) log
	\frac {p(x,y)} {p(x)p(y)}
\end{align*}$$

-	与数据集具体分布有关、与具体取值无关
	-	绝对大小同易受熵影响，（经验）熵较大时，互信息也相对
		较大
	-	由于误差存在，分类取值数目较多者信息增益较大

-	可以衡量定性变量间相关性
	-	信息增益越大，变量之间相关性越强，自变量预测因变量
		能力越强
	-	只能考察特征对整个系统的贡献，无法具体到特征某个取值
	-	只适合作全局特征选择，即所有类使用相同的特征集合

####	*Infomation Gain Ratio*

信息增益比

$$\begin{align*}
g_R(Y|X) & = \frac {g(Y|X)} {H(X)}
\end{align*}$$

-	考虑熵大小，减弱熵绝对大小的影响

###	*Cross Entropy*

> - 信息论：基于相同事件测度的两个概率分布$p, q$，基于非自然
	（相较于真实分布$p$）概率分布$q$进行编码，在事件集合中
	唯一标识事件所需bit
> - 概率论：概率分布$p, q$之间差异

$$\begin{align*}
H(p, q) & = E_p[-log q] = \left \{ \begin{array}{l}
	= -\sum_{x} p(x) logq(x), & 离散分布 \\
	= -\int_X P(x) log(Q(x)) d(r(x)), & 连续分布
\end{array} \right. \\
& = H(p) + D_{KL}(p||q)
\end{align*}$$

> - $q(x)$：离散非自然概率分布
> - $Q(x)$：连续非自然概率分布密度函数
> - $r(x)$：测度，通常是$Borel \sigma$代数上的勒贝格测度
> - $D_{KL}(p||q)$：$p$到$q$的KL散度（$p$相对于$q$的相对熵）

-	交叉熵是常用的损失函数：效果等价于KL散度，但计算方便

> - sigmoid激活函数时：相较于二次损失，收敛速度更快

###	*Kullback-Leibler Divergence*

KL散度/相对熵：概率分布$p, q$之间差异量化指标

$$\begin{align*}
D_{KL}(p||q) & = E_p[log p(x) - log q(x)] \\
& = \sum_{i=1}^N p(x_i) (log p(x_i) - log q(x_i)) \\
& = \sum_{i=1} p(x_i) log \frac {p(x_i)} {q(x_i)}
\end{align*}$$

-	KL散度表示：原始分布$p$、近似分布$q$之间对数差值期望
-	KL散度不对称，分布$p$度量$q$、$q$度量$p$损失信息不同
	-	从计算公式也可以看出
	-	KL散度不能作为不同分布之间距离的度量

##	卡方统计量

卡方统计量：通过观察实际与理论值的偏差确定理论正确与否

$$
\mathcal{X}^2 = \sum \frac {(A - E)^2} E
$$

> - $A$：自变量、因变量组合对应频数观察值
> - $E$：自变量、因变量组合对应频数期望值

-	将模型预测结果视为实际分布、先验分布（均匀分布）视为理论
	分布

-	卡方检验：检验定性变量之间相关性，假设两个变量确实独立，
	观察实际值、理论值偏差程度判断变量之间相关性

	-	若偏差足够小，认为误差是自然的样本误差，两者确实独立
	-	若偏差大到一定程度，误差不可能由偶然、测量精度导致，
		认为两者相关

-	若模型预测结果同先验分布差别很大，说明模型有效，且卡方
	统计量值越大表示预测把握越大

###	特点

-	由于随机误差存在，卡方统计量容易夸大频数较小的特征影响
-	只存在少数类别中特征的卡方统计量值可能很小，容易被排除，
	而往往这类词对分类贡献很大

##	Odds/Odds Ratio

-	*Odds*：几率/优势，事件发生与不发生的概率比值

	$$
	odds = \frac p {1-p}
	$$

	> - $p$：事件发生概率

-	*Odds Ratio*：优势比，两组事件odds的比值

	$$
	OR = \frac {odds_1} {odds_0}
	$$

##	WOE值

WOE值：将预测变量（二分类场景中）集中度作为分类变量编码
的数值

$$\begin{align*}
WOE_i & = log(\frac {\%B_i} {\%G_i}) \\
& = log(\frac {\#B_i / \#B_T} {\#G_i / \#G_T}) \\
& = log(\frac {\#B_i / \#G_i} {\#B_T / \#G_T}) \\
& = log(\frac {\#B_i} {\#G_i}) - log(\frac {\#B_T} {\#G_T}) \\
& = log(\frac {\#B_i / ({\#B_i + \#G_i})}
	{\#G_i / (\#B_i + \#G_i)}) -
	log(\frac {\#B_T} {\#G_T}) \\
& = log(odds_i) - log(odds_T)
\end{align*}$$

> - $\%B_i, \%G_i$：分类变量取第$i$值时，预测变量为B类、G类占
	所有B类、G类比例
> - $\#B_i, \#B_T$：分类变量取第$i$值时，预测变量为B类占所有
	B类样本比例
> - $\#G_i, \#G_T$：分类变量取第$i$值时，预测变量为G类占所有
	G类样本比例
> - $odds_i$：分类变量取第$i$值时，预测变量取B类优势
> - $odds_T$：所有样本中，预测变量取B类优势
> - 其中$log$一般取自然对数


-	WOE值可以衡量分类变量各取值中
	-	B类占所有B类样本比例、G类占所有G类样本比例的差异
	-	B类、G类比例，与所有样本中B类、G类比例的差异

-	WOE值能体现分类变量取值的预测能力，变量各取值WOE值方差
	越大，变量预测能力越强
	-	WOE越大，表明该取值对应的取B类可能性越大
	-	WOE越小，表明该取值对应的取G类可能性越大
	-	WOE接近0，表明该取值预测能力弱，对应取B类、G类可能性
		相近

###	OR与WOE线性性

$$\begin{align*}
log(OR_{j,i}) &= log(odds_i) - log(odds_j) \\
&= WOE_i - WOE_j
\end{align*}$$

-	即：预测变量对数优势值与WOE值呈线性函数关系
	-	预测变量在取$i,j$值情况下，预测变量优势之差为取$i,j$
		值的WOE值之差
	-	WOE值编码时，分类变量在不同取值间跳转时类似于线性
		回归中数值型变量

	![woe_encoding_linear_sketch](imgs/woe_encoding_linear_sketch.png)

-	考虑到对数优势的数学形式，单变量LR模型中分类型变量WOE
	值可以类似数值型变量直接入模
	-	当然，WOE值编码在多元LR中无法保证单变量分类情况下的
		线性性
	-	或者说多变量LR中个变量系数值不一定为1
	-	在基于单变量预测能力优秀在多变量场合也优秀的假设下，
		WOE值编码（IV值）等单变量分析依然有价值

###	Bayes Factor、WOE编码、多元LR

$$\begin{align*}
ln(\frac {P(Y=1|x_1,x_2,\cdots,x_D)}
	{P(Y=0|x_1,x_2,\cdots,x_D)})
	&= ln(\frac {P(Y=1)} {P(Y=0)}) \\
	& \overset {conditionally independent} {=}
		ln (\frac {P(Y=1)} {P(Y=0)}) + 
		\sum_{i=1}^D ln(\frac {P(x_i|Y=1)} {P(x_i|Y=0)}) \\
ln(\frac {P(Y=1|x_1,x_2,\cdots,x_D)} 
	{P(Y=0|x_1,x_2,\cdots,x_D)})
	& \overset {semi} {=} ln (\frac {P(Y=1)} {P(Y=0)}) +
		\sum_{i=1}^D \beta_i ln(\frac {P(x_i|Y=1)}
		{P(x_i|Y=0)})
\end{align*}$$

> - $\frac {P(x_i|Y=1)} {P(x_i|Y=0)}$：贝叶斯因子，常用于
	贝叶斯假设检验

-	*Naive Bayes*中满足各特征$X$关于$Y$条件独立的强假设下，
	第二个等式成立

-	*Semi-Naive Bayes*中放宽各特征关于$Y$条件独立假设，使用
	权重体现变量相关性，此时则可以得到多元LR的预测变量取值
	对数OR形式
	-	则多元LR场景中，WOE值可以从非完全条件独立的贝叶斯
		因子角度理解

###	IV值

$$\begin{align*}
IV_i &= (\frac {\#B_i} {\#B_T} - \frac {\#G_i} {\#G_T}) * 
	WOE_i \\
&= (\frac {\#B_i} {\#B_T} - \frac {\#G_i} {\#G_T}) *
	log(\frac {\#B_i / \#B_T} {\#G_i / \#G_T}) \\
IV &= \sum IV_i
\end{align*}$$

> - $IV_i$：特征$i$取值IV值
> - $IV$：特征总体IV值

-	特征总体的IV值实际上是其各个取值IV值的加权和
	-	类似交叉熵为各取值概率的加权和





