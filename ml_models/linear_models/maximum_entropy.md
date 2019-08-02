---
title: 最大熵模型
tags:
  - 模型
  - 线性模型
categories:
  - 模型
  - 线性模型
date: 2019-08-01 02:10:08
updated: 2019-08-01 02:10:08
toc: true
mathjax: true
comments: true
description: 最大熵模型
---

##	逻辑斯蒂回归

###	逻辑斯蒂分布

$$\begin{align*}
F(x) & = P(X \leq x) = \frac 1 {1 + e^{-(x-\mu)/\gamma}} \\
f(x) & = F^{'}(x) = \frac {e^{-(x-\mu)/\gamma}}
	{\gamma(1+e^{-(x-\mu)/\gamma})^2}
\end{align*}$$

> - $\mu$：位置参数
> - $\gamma$：形状参数

-	分布函数属于逻辑斯蒂函数
-	分布函数图像为sigmoid curve
	-	关于的$(\mu, \frac 1 2)$中心对称
		$$
		F(-x+\mu) - \frac 1 2 = -F(x+\mu) + \frac 1 2
		$$
	-	曲线在靠近$\mu$中心附近增长速度快，两端速度增长慢
	-	形状参数$\gamma$越小，曲线在中心附近增加越快

###	*Binomial Logistic Regression Model*

二项逻辑斯蒂回归模型：形式为参数化逻辑斯蒂分布的二分类
生成模型

$$\begin{align*}
P(Y=1|x) & = \frac {exp(wx + b)} {1 + exp (wx + b)} \\
P(Y=0|x) & = \frac 1 {exp(wx + b)} \\
P(Y=1|\hat x) & = \frac {exp(\hat w \hat x)}
	{1 + exp (\hat w \hat x)} \\
P(Y=0|\hat x) & = \frac 1 {1+exp(\hat w \hat x)}
\end{align*}$$

> - $w, b$：权值向量、偏置
> - $\hat x = (x^T|1)^T$
> - $\hat w = (w^T|b)^T$

-	逻辑回归比较两个条件概率值，将实例x归于条件概率较大类

####	逻辑回归性质

> - *odds*：几率，事件发生与不发生的概率比值$\frac p {1-p}$

-	对逻辑回归有

	$$
	log \frac {P(Y=1|x)} {1-P(Y=0|x)} = \hat w \hat x
	$$

	-	在逻辑回归模型中，输出Y=1的对数几率是输入x的线性函数

-	通过逻辑回归模型，可以将线性函数$wx$转换为概率
	-	线性函数值越接近正无穷，概率值越接近1
	-	线性函数值越接近负无穷，概率值越接近0

####	策略

极大似然：极小对数损失（交叉熵损失）

$$\begin{align*}
L(w) & = log \prod_{i=1}^N [\pi(x_i)]^{y_i}
	[1-\pi(x_i)]^{1-y_i} \\
& = \sum_{i=1}^N [y_i log \pi(x_i) + (1-y_i)log(1-\pi(x_i))] \\
& = \sum_{i=1}^N [y_i log \frac {\pi(x_i)}
	{1-\pi(x_i)} log(1-\pi(x_i))] \\
& = \sum_{i=1}^N [y_i(\hat w \hat x_i) -
	log(1+exp(\hat w \hat x_i))]
\end{align*}$$

> - $\pi(x) = P(Y=1|x)$

####	算法

-	通常采用梯度下降、拟牛顿法求解有以上最优化问题

###	*Multi-Nominal Logistic Regression Model*

多项逻辑斯蒂回归：二项逻辑回归模型推广

$$\begin{align*}
P(Y=j|x) & = \frac {exp(\hat w_j \hat x)} {1+\sum_{k=1}^{K-1}
	exp(\hat w_k \hat x)}, k=1,2,\cdots,K-1 \\
P(Y=K|x) & = \frac 1 {1+\sum_{k=1}^{K-1}
	exp(\hat w_k \hat x)}
\end{align*}$$

-	策略、算法类似二项逻辑回归模型

##	*Generalized Linear Model*

#todo

##	*Maximum Entropy Model*

###	最大熵原理

最大熵原理：学习概率模型时，在所有可能的概率模型（分布）中，
**熵最大的模型是最好的模型**

-	使用约束条件确定概率模型的集合，则最大熵原理也可以表述为
	**在满足约束条件的模型中选取熵最大的模型**

-	直观的，最大熵原理认为
	-	概率模型要满足已有事实（约束条件）
	-	没有更多信息的情况下，不确定部分是等可能的
	-	等可能不容易操作，所有考虑使用**可优化**的熵最大化
		表示等可能性

###	最大熵模型

最大熵模型为生成模型

-	对给定数据集$T=\{(x_1,y_1),\cdots,(x_N,y_N)\}$，联合分布
	P(X,Y)、边缘分布P(X)的经验分布如下

	$$\begin{align*}
	\tilde P(X=x, Y=y) & = \frac {v(X=x, Y=y)} N \\
	\tilde P(X=x) & = \frac {v(X=x)} N
	\end{align*}$$

	> - $v(X=x,Y=y)$：训练集中样本$(x,y)$出频数

-	用如下*feature function* $f(x, y)$描述输入x、输出y之间
	某个事实

	$$f(x, y) = \left \{ \begin{array}{l}
	1, & x、y满足某一事实 \\
	0, & 否则
	\end{array} \right.$$

	-	特征函数关于经验分布$\tilde P(X, Y)$的期望
		$$
		E_{\tilde P} = \sum_{x,y} \tilde P(x,y)f(x,y)
		$$

	-	特征函数关于生成模型$P(Y|X)$、经验分布$\tilde P(X)$
		期望
		$$
		E_P(f(x)) = \sum_{x,y} \tilde P(x)P(y|x)f(x,y)
		$$

-	期望模型$P(Y|X)$能够获取数据中信息，则两个期望值应该相等

	$$\begin{align*}
	E_P(f) & = E_{\tilde P}(f) \\
	\sum_{x,y} \tilde P(x)P(y|x)f(x,y) & =
		\sum_{x,y} \tilde P(x,y)f(x,y)
	\end{align*}$$

	此即作为模型学习的约束条件

	-	此约束是纯粹的关于$P(Y|X)$的约束，只是约束形式特殊，
		需要通过期望关联熵

	-	若有其他表述形式、可以直接带入的、关于$P(Y|X)$约束，
		可以直接使用

> - 满足所有约束条件的模型集合为
	$$
	\mathcal{C} = \{P | E_{P(f_i)} = E_{\tilde P (f_i)},
		i=1,2,\cdots,n \}
	$$
	定义在条件概率分布$P(Y|X)$上的条件熵为
	$$
	H(P) = -\sum_{x,y} \tilde P(x) P(y|x) logP(y|x)
	$$
	则模型集合$\mathcal{C}$中条件熵最大者即为最大是模型

###	策略

最大熵模型的策略为以下约束最优化问题

$$\begin{array}{l}
\max_{P \in \mathcal{C}} & -H(P)=\sum_{x,y} \tilde P(x)
	P(y|x) logP(y|x) \\
s.t. & E_P(f_i) - E_{\tilde P}(f_i) = 0, i=1,2,\cdots,M \\
& \sum_{y} P(y|x)  = 1
\end{array}$$

-	引入拉格朗日函数

	$$\begin{align*}
	L(P, w) & = -H(P) - w_0(1-\sum_y P(y|x)) + \sum_{m=1}^M
		w_m(E_{\tilde P}(f_i) - E_P(f_i)) \\
	& = \sum_{x,y} \tilde P(x) P(y|x) logP(y|x) + w_0
		(1-\sum_y P(y|x)) + \sum_{m=1}^M w_m (\sum_{x,y}
		\tilde P(x,y)f_i(x, y) - \tilde P(x)P(y|x)f_i(x,y))
	\end{align*}$$

	-	原始问题为
		$$
		\min_{P \in \mathcal{C}} \max_{w} L(P, w)
		$$

	-	对偶问题为
		$$
		\max_{w} \min_{P \in \mathcal{C}} L(P, w)
		$$

	-	考虑拉格朗日函数$L(P, w)$是P的凸函数，则原始问题、
		对偶问题解相同

-	记

	$$\begin{align*}
	\Psi(w) & = \min_{P \in \mathcal{C}} L(P, w)
		= L(P_w, w) \\
	P_w & = \arg\min_{P \in \mathcal{C}} L(P, w) = P_w(Y|X)
	\end{align*}$$

-	求$L(P, w)$对$P(Y|X)$偏导

	$$\begin{align*}
	\frac {\partial L(P, w)} {\partial P(Y|X)} & =
		\sum_{x,y} \tilde P(x)(logP(y|x)+1) - \sum_y w_0 -
		\sum_{x,y}(\tilde P(x) \sum_{i=1}^N w_i f_i(x,y)) \\
	& = \sum_{x,y} \tilde P(x)(log P(y|x) + 1 - w_0 -
		\sum_{i=1}^N w_i f_i(x, y))
	\end{align*}$$

	偏导置0，考虑到$\tilde P(x) > 0$，其系数必始终为0，有

	$$\begin{align*}
	P(Y|X) & = \exp(\sum_{i=1}^N w_i f_i(x,y) + w_0 - 1) \\
	& = \frac {exp(\sum_{i=1}^N w_i f_i(x,y))} {exp(1-w_0)}
	\end{align*}$$

-	考虑到约束$\sum_y P(y|x) = 1$，有

	$$\begin{align*}
	P_w(y|x) & = \frac 1 {Z_w(x)} exp(\sum_{i=1}^N w_i
		f_i(x,y)) \\
	Z_w(x) & = \sum_y exp(\sum_{i=1}^N w_i f_i(x,y)) \\
	& = exp(1 - w_0)
	\end{align*}$$

	> - $Z_w(x)$：规范化因子
	> - $f(x, y)$：特征
	> - $w_i$：特征权值

-	原最优化问题等价于求解偶问题极大化问题$\max_w \Psi(w)$

	$$\begin{align*}
	\Psi(w) & = \sum_{x,y} \tilde P(x) P_w(y|x) logP_w(y|x)
		+ \sum_{i=1}^N w_i(\sum_{x,y} \tilde P(x,y) f_i(x,y)
		- \sum_{x,y} \tilde P(x) P_w(y|x) f_i(x,y)) \\
	& = \sum_{x,y} \tilde P(x,y) \sum_{i=1}^N w_i f_i(x,y) +
		\sum_{x,y} \tilde P(x,y) P_w(y|x)(log P_w(y|x) -
		\sum_{i=1}^N w_i f_i(x,y)) \\
	& = \sum_{x,y} \tilde P(x,y) \sum_{i=1}^N w_i f_i(x,y) -
		\sum_{x,y} \tilde P(x,y) P_w(y|x) log Z_w(x) \\
	& = \sum_{x,y} \tilde P(x,y) \sum_{i=1}^N w_i f_i(x,y) -
		\sum_x \tilde P(x) log Z_w(x)
	\end{align*}
	$$
	
	记其解为

	$$w^{*} = \arg\max_w \Psi(w)$$

	带入即可得到最优（最大熵）模型$P_{w^{*}}(Y|X)$

####	策略性质

-	已知训练数据的经验概率分布为$\tilde P(X,Y)$，则条件概率
	分布$P(Y|X)$的对数似然函数为

	$$\begin{align*}
	L_{\tilde P}(P_w) & = N log \prod_{x,y}
		P(y|x)^{\tilde P(x,y)} \\
	& = \sum_{x,y} N * \tilde P(x,y) log P(y|x)
	\end{align*}$$

	> - 这里省略了系数样本数量$N$

-	将最大熵模型带入，可得

	$$\begin{align*}
	L_{\tilde P_w} & = \sum_{x,y} \tilde P(y|x) logP(y|x) \\
	& = \sum_{x,y} \tilde P(x,y) \sum_{i=1}^N w_i f_i(x,y) -
		\sum_{x,y} \tilde P(x,y)log Z_w(x) \\
	& = \sum_{x,y} \tilde P(x,y) \sum_{i=1}^N w_i f_i(x,y) -
		\sum_x \tilde P(x) log Z_w(x) \\
	& = \Psi(w)
	\end{align*}$$

	对偶函数$\Psi(w)$等价于对数似然函数$L_{\tilde P}(P_w)$，
	即最大熵模型中，**对偶函数极大等价于模型极大似然估计**

###	改进的迭代尺度法

-	思想
	-	假设最大熵模型当前参数向量$w=(w_1,w_2,\cdots,w_M)^T$
	-	希望能找到新的参数向量（参数向量更新）
		$w+\sigma=(w_1+\sigma_1,\cdots,w_M+\sigma_M)$
		使得模型对数似然函数/对偶函数值增加
	-	不断对似然函数值进行更新，直到找到对数似然函数极大值

-	对给定经验分布$\tilde P(x,y)$，参数向量更新至$w+\sigma$
	时，对数似然函数值变化为

	$$\begin{align*}
	L(w+\sigma) - L(w) & = \sum_{x,y} \tilde P(x,y)
		log P_{w+\sigma}(y|x) - \sum_{x,y} \tilde P(x,y)
		log P_w(y|x) \\
	& = \sum_{x,y} \tilde P(x,y) \sum_{i=1}^M \sigma_i
		f_i(x,y) - \sum_x \tilde P(x) log \frac
		{Z_{w+\sigma}(x)} {Z_w(x)} \\
	& \geq \sum_{x,y} \tilde P(x,y) \sum_{i=1}^M \sigma_i
		f_i(x,y) + 1 - \sum_x \tilde P(x) \frac
		{Z_{w+\sigma}(x)} {Z_w(x)} \\
	& = \sum_{x,y} \tilde P(x,y) \sum_{i=1}^M \sigma_i
		f_i(x,y) + 1 - \sum_x \tilde P(x) \sum_y P_y(y|x)
		exp(\sum_{i=1}^M \sigma_i f_i(x,y))
	\end{align*}$$

	-	不等式步利用$a - 1 \geq log a, a \geq 1$

	-	最后一步利用

		$$\begin{align*}
		\frac {Z_{w+\sigma}(x)} {Z_w(x)} & = \frac 1 {Z_w(x)}
			\sum_y exp(\sum_{i=1}^M (w_i + \sigma_i)
			f_i(x, y)) \\
		& = \frac 1 {Z_w(x)} \sum_y exp(\sum_{i=1}^M w_i
			f_i(x,y) + \sigma_i f_i(x,y)) \\
		& = \sum_y P_w(y|x) exp(\sum_{i=1}^n \sigma_i
			f_i(x,y))
		\end{align*}$$

-	记上式右端为$A(\sigma|w)$，则其为对数似然函数改变量的
	一个下界

	$$
	L(w+\sigma) - L(w) \geq A(\sigma|w)
	$$

	-	若适当的$\sigma$能增加其值，则对数似然函数值也应该
		增加
	-	函数$A(\sigma|w)$中因变量$\sigma$为向量，难以同时
		优化，尝试每次只优化一个变量$\sigma_i$，固定其他变量
		$\sigma_j$

-	记

	$$f^{**} (x,y) = \sum_i f_i(x,y)$$

	考虑到$f_i(x,y)$为二值函数，则$f^{**}(x,y)$表示所有特征
	在$(x,y)$出现的次数，且有

	$$
	A(\sigma|w) = \sum_{x,y} \tilde P(x,y) \sum_{i=1}^M
		\sigma_i f_i(x,y) + 1 - \sum_x \tilde P(x)
		\sum_y P_w(y|x) exp(f^{**}(x,y) \sum_{i=1}^M
		\frac {\sigma_i f_i(x,y)} {f^{**}(x,y)})
	$$

-	考虑到$\sum_{i=1}^M \frac {f_i(x,y)} {f^{**}(x,y)} = 1$，
	由指数函数凸性、Jensen不等式有

	$$
	exp(\sum_{i=1}^M \frac {f_i(x,y)} {f^{**}(x,y)} \sigma_i
		f^{**}(x,y)) \leq \sum_{i=1}^M \frac {f_i(x,y)}
		{f^{**}(x,y)} exp(\sigma_i f^{**}(x,y))
	$$

	则

	$$
	A(\sigma|w) \geq \sum_{x,y} \tilde P(x,y) \sum_{i=1}^M
		\sigma_i f_i(x,y) + 1 - \sum_x \tilde P(x) \sum_y
		P_w(y|x) \sum_{i=1}^M \frac {f_i(x,y)} {f^{**}(x,y)}
		exp(\sigma_i f^{**}(x,y))
	$$

-	记上述不等式右端为$B(\sigma|w)$，则有

	$$
	L(w+\sigma) - L(w) \geq B(\sigma|w)
	$$

	其为对数似然函数改变量的一个新、相对不紧的下界

-	求$B(\sigma|w)$对$\sigma_i$的偏导

	$$
	\frac {\partial B(\sigma|w)} {\partial \sigma_i} =
		\sum_{x,y} \tilde P(x,y) f_i(x,y) -
		\sum_x \tilde P(x) \sum_y P_w(y|x) f_i(x,y)
		exp(\sigma_i f^{**}(x,y))
	$$

	置偏导为0，可得

	$$
	\sum_x \tilde P(x) \sum_y P_w(y|x) f_i(x,y) exp(\sigma_i
		f^{**}(x,y)) = \sum_{x,y} \tilde P(x,y) f_i(x,y) =
		E_{\tilde P}(f_i)
	$$

	其中仅含变量$\sigma_i$，则依次求解以上方程即可得到
	$\sigma$

####	算法

> - 输入：特征函数$f_1, f_2, \cdots, f_M$、经验分布
	$\tilde P(x)$、最大熵模型$P_w(x)$
> - 输出：最优参数值$w_i^{*}$、最优模型$P_{w^{*}}$

1.	对所有$i \in \{1,2,\cdots,M\}$，取初值$w_i = 0$

2.	对每个$i \in \{1,2,\cdots,M\}$，求解以上方程得$\sigma_i$

	-	若$f^{**}(x,y)=C$为常数，则$\sigma_i$有解析解

		$$
		\sigma_i = \frac 1 C log \frac {E_{\tilde P}(f_i)}
			{E_P(f_i)}
		$$

	-	若$f^{**}(x,y)$不是常数，则可以通过牛顿法迭代求解

		$$
		\sigma_i^{(k+1)} = \sigma_i^{(k)} - \frac
			{g(\sigma_i^{(k)})} {g^{'}(\sigma_i^{(k)})}
		$$

		> - $g(\sigma_i)$：上述方程对应函数

		-	上述方程有单根，选择适当初值则牛顿法恒收敛

3.	更新$w_i$，$w_i \leftarrow w_i + \sigma_i$，若不是所有
	$w_i$均收敛，重复2

###	BFGS算法

对最大熵模型

-	为方便，目标函数改为求极小

	$$\begin{array}{l}
	\min_{w \in R^M} f(w) = \sum_x \tilde P(x) log \sum_{y}
		exp(\sum_{i=1}^M w_i f_i(x,y)) - \sum_{x,y}
		\tilde P(x,y) \sum_{i=1}^M w_i f_i(x,y)
	\end{array}$$

-	梯度为

	$$\begin{align*}
	g(w) & = (\frac {\partial f(w)} {\partial w_i}, \cdots,
		\frac {\partial f(w)} {\partial w_M})^T \\
	\frac {\partial f(w)} {\partial w_M} & = \sum_{x,y}
		\tilde P(x) P_w(y|x) f_i(x,y) - E_{\tilde P}(f_i)
	\end{align*}$$

####	算法

将目标函数带入BFGS算法即可

> - 输入：特征函数$f_1, f_2, \cdots, f_M$、经验分布
	$\tilde P(x)$、最大熵模型$P_w(x)$
> - 输出：最优参数值$w_i^{*}$、最优模型$P_{w^{*}}$

1.	取初值$w^{(0)}$、正定对称矩阵$B^{(0)}$，置k=0

2.	计算$g^{(k)} = g(w^{(k)})$，若$\|g^{(k)}\| < \epsilon$，
	停止计算，得到解$w^{*} = w^{(k)}$

3.	由拟牛顿公式$B^{(k)}p^{(k)} = -g^{(k)}$求解$p^{(k)}$

4.	一维搜索，求解

	$$
	\lambda^{(k)} = \arg\min_{\lambda} f(w^{(k)} +
		\lambda p_k)
	$$

5.	置$w^{(k+1)} = w^{(k)} + \lambda^{(k)} p_k$

6.	计算$g^{(k+1)} = g(w^{(k+1)})$，若
	$\|g^{(k+1)}\| < \epsilon$，停止计算，得到解
	$w^{*} = w^{(k+1)}$，否则求

	$$
	B^{(k+1)} = B^{(k)} - \frac {B^{(k)} s^{(k)}
		(s^{(k)})^T B^{(k)}} {(s^{(k)})^T B^{(k)} s^{(k)}}
		+ \frac {y^{(k)} (y^{(k)})^T} {(y^{(k)})^T s^{(k)}}
	$$

	> - $s^{(k)} = w^{(k+1)} - w^{(k)}$
	> - $y^{(k)} = g^{(k+1)} - g^{(k)}$

7.	置k=k+1，转3


