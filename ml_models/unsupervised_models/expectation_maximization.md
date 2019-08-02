---
title: EM算法
tags:
  - 模型
  - 无监督模型
categories:
  - 模型
  - 无监督模型
date: 2019-07-13 23:26:35
updated: 2019-07-13 12:03:12
toc: true
mathjax: true
comments: true
description: EM算法
---

##	总述

*expectation maximization algorithm*：含有隐变量的概率模型
参数的极大似然估计法、极大后验概率估计法

-	模型含有*latent variable*（潜在变量）、*hidden variable*
	（隐变量）似然函数将没有解析解
	
-	所以EM算法需要迭代求解，每次迭代由两步组成
	-	E步：求期望expectation
	-	M步：求极大maximization

> - 模型变量都是*observable variable*、给定数据情况下，可以
	直接使用极大似然估计、贝叶斯估计

##	EM算法

对含有隐变量的概率模型，目标是极大化观测数据（不完全数据）
$Y$关于参数$\theta$的对数似然函数，即极大化

$$\begin{align*}
L(\theta) & = log P(Y|\theta) \\
& = log \sum_Z P(Y, Z|\theta) \\
& = log \left(\sum_Z P(Y|Z,\theta) P(Z|\theta) \right)
\end{align*}$$

> - $Y$：观测变量数据
> - $Z$：隐随机变量数据（未知）
> - $Y,Z$合在一起称为完全数据
> - $P(Y,Z|\theta)$：联合分布
> - $P(Z|Y,\theta)$：条件分布

-	但是极大化目标函数中包括未观测数据$Z$、求和（积分）的
	对数，直接求极大化非常困难
-	EM算法通过**迭代**逐步近似极大化$L(\theta)$

###	推导

-	假设第i次迭代后$\theta$的估计值是$\theta^{(i)}$，希望
	新估计值$\theta$能使$L(\theta)$增加，并逐步增加到极大值
	，考虑两者之差

	$$
	L(\theta) - L(\theta^{(i)}) = log (\sum_Z P(Y|Z,\theta)
		P(Z|\theta)) - log P(Y|\theta^{(i)})
	$$

-	利用Jensen不等式有

	$$\begin{align*}
	L(\theta) - L(|theta^{(i)}) & = log(\sum_Z P(Y|Z,
		\theta^{(i)}) \frac {P(Y|Z,\theta) P(Z|\theta)}
		{P(Y|Z,\theta^{(i)})}) - log P(Y|\theta^{(i)}) \\
	& \geq \sum_Z P(Z|Y,\theta^{(i)}) log \frac
		{P(Y|Z,\theta) P(Z|\theta)} {P(Z|Y,\theta^{(i)})}
		- log P(Y|\theta^{(i)}) \\
	& = \sum_z P(Z|Y,\theta^{(i)}) log \frac
		{P(Y|Z,\theta) P(Z|\theta)}
		{P(Z|Y,\theta^{(i)}) P(Y|\theta^{(i)})}
	\end{align*}$$

-	令

	$$
	B(\theta, \theta^{(i)}) = L(\theta^{(i)}) + \sum_Z
		P(Z|Y,\theta^{(i)}) log \frac
		{P(Y|Z,\theta) P(Z|\theta)}
		{P(Z|Y,\theta^{(i)}) P(Y|\theta^{(i)})}
	$$

	则$B(\theta, \theta^{(i)})$是$L(\theta)$的一个下界，即

	$$\begin{align*}
	L(\theta) & \geq B(\theta, \theta^{(i)}) \\
	\end{align*}$$

	并根据$B(\theta, \theta^{(i)})$定义有b

	$$\begin{align*}
	L(\theta^{(i)}) = B(\theta^{(i)}, \theta^{(i)})
	\end{align*}$$

-	则任意$\theta$满足
	$B(\theta,\theta^{(i)}) > B(\theta^{(i)},\theta^{(i)})$
	，将满足$L(\theta) > L(\theta^{(i)})$，应选择
	$\theta^{(i+1)}$使得$B(\theta,\theta^{(i)})$达到极大

	$$\begin{align*}
	\theta^{(i+1)} & = \arg\max_{\theta}
		B(\theta,\theta^{(i)}) \\
	& = \arg\max_{\theta} L(\theta^{(i)}) + \sum_Z
		P(Z|Y,\theta^{(i)}) log \frac
		{P(Y|Z,\theta) P(Z|\theta)}
		{P(Z|Y,\theta^{(i)}) P(Y|\theta^{(i)})} \\
	& = \arg\max_{\theta} (\sum_Z P(Z|Y,\theta^{(i)})
		log(P(Y|Z,\theta)P(Z|\theta))) \\
	& = \arg\max_{\theta} (\sum_Z P(Z|Y,\theta^{(i)})
		log P(Y,Z|\theta)) \\
	& = \arg\max_{\theta} Q(\theta, \theta^{(i)})
	\end{align*}
	$$

	> - 和$\theta$无关的常数项全部舍去

> - $Q(\theta, \theta^{(i)})$：Q函数，完全数据的对数似然函数
	$logP(Y,Z|\theta)$，关于在给定观测$Y$和当前参数
	$\theta^{(i)}$下，对未观测数据Z的条件概率分布
	$P(Z|Y,\theta^{(i)})$
	$$
	Q(\theta, \theta^{(i)}) = E_z
		[logP(Y,Z|\theta)|Y,\theta^{(i)}]
	$$

###	算法

1.	选择参数初值$\theta^{0}$，开始迭代

2.	E步：记$\theta^{(i)}$为第$i$迭代时，参数$\theta$的估计值
	，在第$i+1$步迭代的E步时，计算Q函数
	$Q(\theta, \theta^{(i)})$

3.	M步：求使得Q函数极大化$\theta$作为第$i+1$次估计值
	$\theta^{(i+1)}$

	$$
	\theta^{(i+1)} = \arg\max_{\theta} Q(\theta, \theta^{(i)})
	$$

4.	重复E步、M步直到待估参数收敛

> - 算法初值可以任意选择，但EM算法对初值敏感

> - E步：参数值估计缺失值分布，计算Q函数（似然函数）

> - M步：Q函数取极大得新参数估计值

> - 收敛条件一般是对较小正数$\epsilon$，满足
	$\|\theta^{(i+1)} - \theta^{(i)}\| < \epsilon$或
	$\|Q(\theta^{(i+1)},\theta^{(i)}) - Q(\theta^{(i)},\theta^{(i)})\| < \epsilon$

###	EM算法特点

####	EM算法优点

-	EM算法可以用于估计含有隐变量的模型参数
-	非常简单，稳定上升的步骤能非常可靠的找到最优估计值
-	应用广泛，能应用在多个领域中
	-	生成模型的非监督学习

####	EM算法缺点

-	EM算法计算复杂、受外较慢，不适合高维数据、大规模数据集
-	参数估计结果依赖初值，不够稳定，不能保证找到全局最优解

###	算法收敛性

####	定理1

> - 设$P(Y|\theta)$为观测数据的似然函数，$\theta^{(i)}$为
	EM算法得到的参数估计序列，$P(Y|\theta^{(i)}),i=1,2,...$
	为对应的似然函数序列，则$P(Y|\theta^{(i)})$是单调递增的
	$$
	P(Y|\theta^{(i+1)}) \geq P(Y|\theta^{(i)})
	$$

-	由条件概率

	$$\begin{align*}
	P(Y|\theta) & = \frac {P(Y,Z|\theta)} {P(Z|Y,\theta)} \\
	logP(Y|\theta) & = logP(Y,Z|\theta) - logP(Z|Y,\theta)
	\end{align*}$$


-	则对数似然函数有

	$$
	logP(Y|\theta) = Q(\theta, \theta^{(i)}) -
		H(\theta, \theta^{(i)})
	$$

	> - $H(\theta, \theta^{(i)}) = \sum_Z log P(Z|Y,\theta) P(Z|Y,\theta)$
	> - $Q(\theta, \theta^{(i)})$：前述Q函数
	> - $logP(Y|\theta)$和$Z$无关，可以直接提出

-	分别取$\theta^{(i+1)}, \theta^{(i)}$带入，做差

	$$
	logP(Y|\theta^{(i+1)}) - logP(Y|\theta^{(i)}) =
		[Q(\theta^{(i+1)}, \theta^{(i)}) - 
		Q(\theta^{(i)}, \theta^{(i)}] -
		[H(\theta^{(i+1)}, \theta^{(i)}) -
		H(\theta^{(i)}, \theta^{(i)})]
	$$

	-	$\theta^{(i+1)}$使得$Q(\theta, \theta^{(i)})$取极大

	-	又有

		$$\begin{align*}
		& H(\theta^{(i+1)}, \theta^{(i)}) -
			H(\theta^{(i)}, \theta^{(i)}) \\
		= & \sum_Z (log \frac {P(Z|Y,\theta^{(i+1)})}
			{P(Z|Y,\theta^{(I)})}) P(Z|Y,\theta^{(i)}) \\
		\leq & log (\sum_Z \frac {P(Z|Y,\theta^{(i+1)})}
			{P(Z|Y,\theta^{(I)})} P(Z|Y,\theta^{(i)})) \\
		= & log \sum_Z P(Z|Y,\theta^{(i+1)}) = 0
		\end{align*}$$

####	定理2

> - 设$L(\theta)=log P(Y|\theta)$为观测数据的对数似然函数，
	$\theta^{(i)},i=1,2,...$为EM算法得到的参数估计序列，
	$L(\theta^{(i)}),i=1,2,...$为对应的对数似然函数序列
> > -	若$P(Y|\theta)$有上界，则$L(\theta^{(i)})$收敛到某
		定值$L^{*}$
> > -	Q函数$Q(\theta, \theta^{'})$与$L(\theta)$满足一定
		条件的情况下，由EM算法得到的参数估计序列
		$\theta^{(i)}$的收敛值$\theta^{*}$是$L(\theta)$的
		稳定点

-	结论1由序列单调、有界显然

> - Q函数$Q(\theta, \theta^{'})$与$L(\theta)$的条件在大多数
	情况下是满足的
> - EM算法收敛性包含对数似然序列$L(\theta^{(i)})$、参数估计
	序列$\theta^{(i)}$的收敛性，前者不蕴含后者
> - 此定理只能保证参数估计序列收敛到对数似然序列的稳定点，
	不能保证收敛到极大点，可选取多个不同初值迭代，从多个结果
	中选择最好的

##	*Gaussion Mixture Model*

> - 高斯混合模型是指具有如下概率分布模型
	$$
	P(y|\theta) = \sum_{k=1}^K \alpha_k \phi(y|\theta_k)
	$$
> > -	$\alpha_k \geq 0, \sum_{k=1}^K \alpha_k=1$：系数
> > -	$\phi(y|\theta_k)$：高斯分布密度函数
> > -	$\theta_k=(\mu_k, \sigma_k)$：第k个分模型参数

-	用EM算法估计高斯混合模型参数
	$\theta=(\alpha_1,...,\alpha_2,\theta_1,...,\theta_K)$

###	推导

####	明确隐变量

明确隐变量，写出完全数据对数似然函数

-	反映观测数据$y_j$来自第k个分模型的数据是未知的

	$$\gamma_{j,k} = \left \{ \begin{array}{l}
	1, & 第j个观测来自第k个分模型 \\
	0, & 否则
	\end{array} \right.
	$$

	> - $j=1,2,\cdots,N$：观测编号
	> - $k=1,2,\cdots,K$：模型编号

-	则完全数据为

	$$(y_j,\gamma_{j,1},\cdots,\gamma_{j,K}), j=1,2,...,N$$

-	完全数据似然函数为

	$$\begin{align*}
	P(y,\gamma|\theta) & = \prod_{j=1}^N
		P(y_j,\gamma_{j,1},\cdots,\gamma_{j,N}|\theta) \\
	& = \prod_{k=1}^{K} \prod_{j=1}^N
		[\alpha_k \phi(y_j|\theta_k)]^{\gamma _{j,k}} \\
	& = \prod_{k=1}^{K} \alpha_k^{n_k} \prod_{j=1}^N
		[\phi(y_j|\theta_k)]^{\gamma _{j,k}} \\
	\end{align*}$$

	> - $n_k = \sum_{j=1}^{N} \gamma_{j,k}$
	> - $\sum_{k=1}^K n_k = N$

-	完全数据的对数似然函数为

	$$
	logP(y, \gamma|\theta) = \sum_{k=1}^K \left \{
		n_k log \alpha_k + \sum_{j=1}^N \gamma_{j,k}
		[log \frac 1 {\sqrt {2\pi}} - log \sigma_k -
		\frac 1 {2\sigma_k}(y_j - \mu_k)^2] \right \}
	$$

####	E步：确定Q函数

$$\begin{align*}
Q(\theta, \theta^{(i)}) & =
	E_z[logP(y,\gamma|\theta)|Y,\theta^{(i)}] \\
& = E \sum_{k=1}^K \left \{ n_k log\alpha_k + \sum_{j=1}^N
	\gamma_{j,k} [log \frac 1 {\sqrt {2\pi}} - log \sigma_k
	- \frac 1 {2\sigma_k}(y_j - \mu_k)^2] \right \} \\
& = \sum_{k=1}^K \left \{ \sum_{k=1}^K (E\gamma_{j,k})
	log\alpha_k + \sum_{j=1}^N (E\gamma_{j,k})
	[log \frac 1 {\sqrt {2\pi}} - log \sigma_k
	- \frac 1 {2\sigma_k}(y_j - \mu_k)^2] \right \}
\end{align*}$$

> - $E\gamma_{j,k} = E(\gamma_{j,k}|y,\theta)$：记为
	$\hat \gamma_{j,k}$

$$\begin{align*}
\hat \gamma_{j,k} & = E(\gamma_{j,k}|y,\theta) =
	P(\gamma_{j,k}|y,\theta) \\
& = \frac {P(\gamma_{j,k}=1, y_j|\theta)}
	{\sum_{k=1}^K P(\gamma_{j,k}=1,y_j|\theta)} \\
& = \frac {P(y_j|\gamma_{j,k}=1,\theta)
	P(\gamma_{j,k}=1|\theta)} {\sum_{k=1}^K
	P(y_j|\gamma_{j,k}=1,\theta) P(\gamma_{j,k}|\theta)} \\
& = \frac {\alpha_k \phi(y_j|\theta _k)}
	{\sum_{k=1}^K \alpha_k \phi(y_j|\theta_k)}
\end{align*}$$

带入可得

$$
Q(\theta, \theta^{(i)}) = \sum_{k=1}^K \left\{
	n_k log\alpha_k + \sum_{k=1}^N \hat \gamma_{j,k}
	[log \frac 1 {\sqrt{2\pi}} - log \sigma_k -
	\frac 1 {2\sigma^2}(y_j - \mu_k)^2] \right \}
$$

####	M步

求新一轮模型参数
$\theta^{(i+1)}=(\hat \alpha_1,...,\hat \alpha_2,\hat \theta_1,...,\hat \theta_K)$

$$\begin{align*}
\theta^{(i+1)} & = \arg\max_{\theta} Q(\theta,\theta^{(i)}) \\
\hat \mu_k & = \frac {\sum_{j=1}^N \hat \gamma_{j,k} y_j}
	{\sum_{j=1}^N \hat \gamma_{j,k}} \\
\hat \sigma_k^2 & = \frac {\sum_{j=1}^N \hat \gamma_{j,k}
	(y_j - \mu_p)^2} {\sum_{j=1}^N \hat \gamma_{j,k}} \\
\hat \alpha_k & = \frac {n_k} N = \frac {\sum_{j=1}^N
	\hat \gamma_{j,k}} N
\end{align*}$$

> - $\hat \theta_k = (\hat \mu_k, \hat \sigma_k^2)$：直接求
	偏导置0即可得
> - $\hat \alpha_k$：在$\sum_{k=1}^K \alpha_k = 1$条件下求
	偏导置0求得

###	算法

> - 输入：观测数据$y_1, y_2,\cdots, y_N$，N个高斯混合模型
> - 输出：高斯混合模型参数

1.	取参数初始值开始迭代

2.	E步：依据当前模型参数，计算分模型k对观测数据$y_j$响应度
	$$
	\hat \gamma_{j,k} = \frac {\alpha \phi(y_k|\theta_k)}
		{\sum_{k=1}^N \alpha_k \phi(y_j|\theta)}
	$$

3.	M步：计算新一轮迭代的模型参数
	$\hat mu_k, \hat \sigma_k^2, \hat \alpha_k$

4.	重复2、3直到收敛

> - GMM模型的参数估计的EM算法非常类似K-Means算法
> > -	E步类似于K-Means中计算各点和各聚类中心之间距离，不过
		K-Means将点归类为离其最近类，而EM算法则是算期望
> > -	M步根据聚类结果更新聚类中心

##	GEM

###	*Maximization-Maximization Algorithm*

####	*Free Energy*函数

> - 假设隐变量数据Z的概率分布为$\tilde P(Z)$，定义分布
	$\tilde P$与参数$\theta$的函数$F(\tilde P, \theta)$如下
	$$
	F(\tilde P, \theta) = E_{\tilde P}
		[log P(Y,Z|\theta)] + H(\tilde P)
	$$

> > -	$H(\tilde P)=-E_{\tilde P} log \tilde P(Z)$：分布
		$\tilde P(Z)$的熵
> > -	通常假设$P(Y,Z|\theta)$是$\theta$的连续函数，则函数
		$F(\tilde P,\theta)$是$\tilde P, \theta$的连续函数

####	定理1

> - 对于固定$\theta$，存在唯一分布$\tilde P_\theta$，极大化
	$F(\tilde P, \theta)$，这时$\tilde P_\theta$由下式给出
	$$
	\tilde P_\theta(Z) = P(Z|Y,\theta)
	$$
	并且$\tilde P_{\theta}$随$\theta$连续变化

-	对于固定的$\theta$，求使得$F(\tilde P, \theta)$的极大，
	构造Lagrange函数

	$$
	L(\tilde P, \lambda, \mu) = F(\tilde P, \theta) +
		\lambda(1 - \sum_Z \tilde P(Z)) - \mu \tilde P(Z)
	$$

	因为$\tilde P(Z)$是概率密度，自然包含两个约束

	$$\left \{ \begin{array}{l}
	\sum_Z \tilde P(Z) = 1 \\
	\tilde P(Z) \geq 0
	\end{array} \right.$$

	即Lagrange方程中后两项

-	对$\tilde P(Z)$求偏导，得

	$$
	\frac {\partial L} {\partial \tilde P(Z)} =
		log P(Y,Z|\theta) - log \tilde P(Z) - \lambda - \mu
	$$

	令偏导为0，有

	$$\begin{align*}
	log P(Y,Z|\theta) - log \tilde P(Z) & = \lambda + \mu \\
	\frac {P(Y,Z|\theta)} {\tilde P(Z)} & = e^{\lambda + \mu}
	\end{align*}$$

-	则使得$F(\tilde P, \theta)$极大的$\tilde P_\theta(Z)$
	应该和$P(Y,Z|\theta)$成比例，由概率密度自然约束有

	$$\tilde P_\theta(Z) = P(Y,Z|\theta) $$

	而由假设条件，$P(Y,Z|\theta)$是$\theta$的连续函数

> - 这里概率密度函数$\tilde P(Z)$是作为自变量出现

> - 理论上对$\tilde P(Z)$和一般的**复合函数求导**没有区别，
	但$E_{\tilde P}, \sum_Z$使得整体看起来非常不和谐
	$$\begin{align*}
	E_{\tilde P} f(Z) & = \sum_Z f(Z) \tilde P(Z) \\
	& = \int f(Z) d(\tilde P(Z))
	\end{align*}$$

####	定理2

> - 若$\tilde P_\theta(Z) = P(Z|Y, \theta)$，则
	$$
	F(\tilde P, \theta) = log P(Y|\theta)
	$$

####	定理3

> - 设$L(\theta)=log P(Y|\theta)$为观测数据的对数似然函数，
	$\theta^{(i)}, i=1,2,\cdots$为EM算法得到的参数估计序列，
	函数$F(\tilde P,\theta)$如上定义
> > -	若$F(\tilde P,\theta)$在$\tilde P^{*}, \theta^{*}$
		上有局部极大值，则$L(\theta)$在$\theta^{*}$也有局部
		最大值
> > -	若$F(\tilde P,\theta)$在$\tilde P^{*}, \theta^{*}$
		达到全局最大，则$L(\theta)$在$\theta^{*}$也达到全局
		最大

-	由定理1、定理2有

	$$
	L(\theta) = logP(Y|\theta) = F(\tilde P_\theta, \theta)
	$$

	特别的，对于使$F(\tilde P,\theta)$极大$\theta^{8}$有

	$$
	L(\theta^{*}) = logP(Y|\theta^{*}) =
		F(\tilde P_\theta^{*}, \theta{*})
	$$

-	由$\tilde P_\theta$关于$\theta$连续，局部点域内不存在点
	$\theta^{**}$使得$L(\theta^{**}) > L(\theta^{*})$，否则
	与$F(\tilde P, \theta^{*})$矛盾

####	定理4

> - EM算法的依次迭代可由F函数的极大-极大算法实现
> - 设$\theta^{(i)}$为第i次迭代参数$\theta$的估计，
	$\tilde P^{(i)}$为第i次迭代参数$\tilde P$的估计，在第
	i+1次迭代的两步为
> > -	对固定的$\theta^{(i)}$，求$\tilde P^{(i)}$使得
		$F(\tilde P, \theta^{(i)})$极大
> > -	对固定的$\tilde P^{(i+1)}$，求$\theta^{(i+1)}$使
		$F(\tilde P^{(t+1)}, \theta)$极大化

-	固定$\theta^{(i)}$

	$$\begin{align*}
	F(\tilde P^{(i+1)}, \theta^{(i)} & = E_{\tilde P^{(t+1)}}
		[log P(Y,Z|\theta)] + H(\tilde P^{(i+1)}) \\
	& = \sum_Z log P(Y,Z|\theta) P(Z|Y,\theta^{(i)}) +
		H(\tilde P^{(i+1)}) \\
	& = Q(\theta, \theta^{(i)}) + H(\tilde P^{(i+1)})
	\end{align*}$$

-	则固定$\tilde P^{(i+1)}$求极大同EM算法M步

###	GEM算法

> - 输入：观测数据，F函数
> - 输出：模型参数

1.	初始化$\theta^{(0)}$，开始迭代

2.	第i+1次迭代：记$\theta^{(i)}$为参数$\theta$的估计值，
	$\tilde P^{(i)}$为函数$\tilde P$的估计，求
	$\tilde P^{(t+1)}$使$\tilde P$极大化$F(\tilde P,\theta)$

3.	求$\theta^{(t+1)}$使$F(\tilde P^{(t+1)l}, \theta)$极大化

4.	重复2、3直到收敛

###	次优解代替最优解

> - 输入：观测数据，Q函数
> - 输出：模型参数

1.	初始化参数$\theta^{(0)}$，开始迭代

2.	第i+1次迭代，记$\theta^{(i)}$为参数$\theta$的估计值，
	计算

	$$\begin{align*}
	Q(\theta, \theta^{(i)}) & = E_Z [
		log P(Y,Z|\theta)|Y,\theta^{(i)}] \\
	& = \sum_Z P(Z|Y, \theta^{(i)}) log P(Y,Z|\theta)
	\end{align*}$$

3.	求$\theta^{(i+1)}$使

	$$
	Q(\theta^{(i+1)}, \theta^{(i)}) >
		Q(\theta^{(i)}, \theta^{(i)})
	$$

4.	重复2、3直到收敛


> - 有时候极大化$Q(\theta, \theta^{(i)})$非常困难，此算法
	仅寻找使目标函数值上升方向

###	ADMM求次优解

> - 输入：观测数据，Q函数
> - 输出：函数模型

1.	初始化参数
	$\theta^{(0)} = (\theta_1^{(0)},...,\theta_d^{(0)})$，
	开始迭代

2.	第i次迭代，记
	$\theta^{(i)} = (\theta_1^{(i)},...,\theta_d^{(i)})$，
	为参数$\theta = (\theta_1,...,\theta_d)$的估计值，计算

	$$\begin{align*}
	Q(\theta, \theta^{(i)}) & = E_Z [
		log P(Y,Z|\theta)|Y,\theta^{(i)}] \\
	& = \sum_Z P(Z|Y, \theta^{(i)}) log P(Y,Z|\theta)
	\end{align*}$$

3.	进行d次条件极大化

	1.	在$\theta_1^{(i)},...,\theta_{j-1}^{(i)},\theta_{j+1}^{(i)},...,\theta_d^{(i)}$
		保持不变条件下
		，求使$Q(\theta, \theta^{(i)})$达到极大的
		$\theta_j^{(i+1)}$

	2.	j从1到d，进行d次条件极大化的，得到
		$\theta^{(i+1)} = (\theta_1^{(i+1)},...,\theta_d^{(i+1)})$
		使得

		$$
		Q(\theta^{(i+1)}, \theta^{(i)}) >
			Q(\theta^{(i)}, \theta^{(i)})
		$$

4.	重复2、3直到收敛


