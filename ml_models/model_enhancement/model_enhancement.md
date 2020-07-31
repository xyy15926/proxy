---
title: Model Enhancement
tags:
  - Model
  - Model Enhancement
  - Machine Learning
categories:
  - ML Model
  - Model Enhancement
date: 2019-07-21 00:46:35
updated: 2019-07-21 00:46:35
toc: true
mathjax: true
comments: true
description: Model Enhancement
---

##	Emsemble Learning

> - 集成学习：训练多个基模型，并将其组合起来，以达到更好的
	预测能力、泛化能力、稳健性
> - *base learner*：基模型，基于**独立样本**建立的、一组
	**具有相同形式**的模型中的一个
> - 组合预测模型：由基模型组合，即集成学习最终习得模型

-	源于样本均值抽样分布思路
	-	$var(\bar{X}) = \sigma^2 / n$
	-	基于独立样本，建立一组具有相同形式的基模型
	-	预测由这组模型共同参与
	-	组合预测模型稳健性更高，类似于样本均值抽样分布方差
		更小

-	关键在于
	-	获得多个独立样本的方法
	-	组合多个模型的方法

###	分类

-	*homogenous ensemble*：同源集成，基学习器属于同一类型
		-	*bagging*
		-	*boosting*

-	*heterogenous ensemble*：异源集成，基学习器不一定属于同
	一类型
	-	*[genralization] stacking*

||Target|Data|parallel|Classifier|Aggregation|
|-----|-----|-----|-----|-----|
|Bagging|减少方差|基于boostrap随机抽样，抗异常值、噪声|模型间并行|同源不相关基学习器，一般是树|分类：投票、回归：平均|
|Boosting|减少偏差|基于误分分步|模型间串行|同源若学习器|加权投票|
|Stacking|减少方差、偏差|K折交叉验证数据、基学习器输出|层内模型并行、层间串行|异质强学习器|元学习器|

> - 以上都是指原始版本、主要用途

###	Boosting

提升方法：将弱可学习算法**提升**为强可学习算法的组合元算法

-	属于加法模型：即基函数的线性组合
-	各模型之间存在依赖关系

![boosting](imgs/boosting.png)

####	分类Boosting

> - **依次**学习多个基分类器
> - 每个基分类器**依之前分类结果调整权重**
> - **堆叠**多个分类器提高分类准确率

-	boosting通过组合多个误分率略好于随机猜测的分类器得到
	误分率较小的分类器，因此boosting适合这两类问题

	-	个体之间难度有很大不同，boosting能够更加关注较难的
		个体
	-	学习器对训练集敏感，boosting驱使学习器在趋同的、
		“较难”的分布上学习，此时boosting就和bagging一样能够
		使得模型更加稳健（但原理不同）

-	boosting能减小预测方差、偏差、过拟合

	-	直觉上，使用在不同的样本上训练的基学习器加权组合，
		本身就能减小学习器的随机变动

	-	基于同样的理由，boosting同时也能减小偏差

	-	过拟合对集成学习有些时候有正面效果，其带来多样性，
		使模型泛化能力更好，前提是样本两足够大，否则小样本
		仍然无法提供多样性

####	回归Boosting

> - **依次**训练多个基学习器
> - 每个基学习器以**之前学习器拟合残差**为目标
> - **堆叠**多个学习器减少整体损失

-	boosting组合模型整体损失（结构化风险）

	$$
	R_{srm} = \sum_{i=1}^N l(y_i, \hat y_i) +
		\sum_{t=1}^M \Omega(f_t)
	$$

	> - $l$：损失函数
	> - $f_t$：基学习器
	> - $\Omega(f_t)$：单个基学习器的复杂度罚
	> - $N, M$：样本数目、学习器数目

-	基学习器损失

	$$
	obj^{(t)} = \sum_{i=1}^N l(y_i, \hat y_i^{(t)}) +
		\Omega(f_t)
	$$

####	最速下降法

使用线性函数拟合$l(y_i, \hat y_i^{(t)})$

$$\begin{align*}
obj^{(t)} & = \sum_i^N l(y_i, \hat y_i^{(t-1)} + f_t(x_i)) +
	\Omega(f_t) \\
& \approx \sum_{i=1}^N [l(y_i, \hat y^{(t-1)}) + g_i f_t(x_i)]
	+ \Omega(f_t)
\end{align*}$$

> - $g_i = \partial_{\hat y} l(y_i, \hat y^{t-1})$

-	一次函数没有极值
-	将所有样本损失视为向量（学习器权重整体施加），则负梯度
	方向损失下降最快，考虑使用负梯度作为伪残差

####	Newton法

使用二次函数拟合$l(y_i, \hat y_i^{(t)}$

$$\begin{align*}
obj^{(t)} & = \sum_i^N l(y_i, \hat y_i^{(t-1)} + f_t(x_i)) +
	\Omega(f_t) \\
& \approx \sum_{i=1}^N [l(y_i, \hat y^{(t-1)}) + g_i f_t(x_i)
	+ \frac 1 2 h_i f_t^2(x_i)] + \Omega(f_t) \\
\end{align*}$$

> - $h_i = \partial^2_{\hat y} l(y_i, \hat y^{t-1})$

-	二次函数本身有极值
-	可以结合复杂度罚综合考虑，使得每个基学习器损失达到最小

###	Boosting&Bagging

-	基分类器足够简单时，boosting表现均显著好于bagging
	-	仅靠单次决策（单个属性、属性组合）分类

-	使用C4.5树作为基分类器时，boosting仍然具有优势，但是不够
	有说服力

> - 结论来自于*Experiments with a New Boosting Algorithm*


####	Boosting&Bagging

-	基分类器足够简单时，boosting表现均显著好于bagging
	-	仅靠单次决策（单个属性、属性组合）分类

-	使用C4.5树作为基分类器时，boosting仍然具有优势，但是不够
	有说服力

> - 结论来自于*Experiments with a New Boosting Algorithm*

###	原理

*probably approximately correct*：概率近似正确，在概率近似
正确学习的框架中

-	*strongly learnable*：强可学习，一个概念（类），如果存在
	一个多项式的学习算法能够学习它，并且**正确率很高**，那么
	就称为这个概念是强可学习的

-	*weakly learnable*：弱可学习，一个概念（类），如果存在
	一个多项式的学习算法能够学习它，学习的正确率仅比随机猜测
	略好，称此概念为弱可学习的

-	*Schapire*证明：在PAC框架下强可学习和弱可学习是等价的

###	具体措施

> - 弱学习算法要比强学习算法更容易寻找，所以具体实施提升就是
	需要解决的问题

-	**改变训练数据权值、概率分布的方法**
	-	提高分类错误样本权值、降低分类正确样本权值

-	**将弱学习器组合成强学习器的方法**
	-	*competeing*
	-	*simple majority voting*
	-	*weighted majority voting*
	-	*confidence-based weighting*

###	学习器组合方式

> - 很多模型无法直接组合，只能组合预测结果

-	*simple majority voting*/*simple average*：简单平均
	$$
	h = \frac 1 K \sum_{k=1}_K h_k
	$$

	> - $h_k$：第k个预测

-	*weighted majority voting*/*weighted average*：加权平均
	$$
	h = \frac {\sum_{k=1}^K w_k h_k} {\sum_{k=1}^K w_k}
	$$

	> - $w_k$：第k个预测权重，对分类器可以是准确率

-	*competing voting*/*largest*：使用效果最优者

-	*confidence based weighted*：基于置信度加权
	$$\begin{align*}
	h = \arg\max_{y \in Y} \sum_{k=1}^K ln(\frac {1 - e_k}
		{e_k}) h_k
	\end{align*}
	$$

	> - $e_k$：第k个模型损失

##	Meta Learning

元学习：自动学习关于关于机器学习的元数据的机器学习子领域

-	元学习主要目标：使用学习到元数据解释，自动学习如何
	*flexible*的解决学习问题，借此提升现有学习算法性能、
	学习新的学习算法，即学习学习

-	学习算法灵活性即可迁移性，非常重要
	-	学习算法往往基于某个具体、假象的数据集，有偏
	-	学习问题、学习算法有效性之间的关系没有完全明白，对
		学习算法的应用有极大限制

###	要素

-	元学习系统必须包含子学习系统
-	学习经验通过提取元知识获得经验，元知识可以在先前单个
	数据集，或不同的领域中获得
-	学习*bias*（影响用于模型选择的前提）必须动态选择
	-	*declarative bias*：声明性偏见，确定假设空间的形式
		，影响搜索空间的大小
		-	如：只允许线性模型
	-	*procedural bias*：过程性偏见，确定模型的优先级
		-	如：简单模型更好

###	*Recurrent Neural networks*

RNN：*self-referential* RNN理论上可以通过反向传播学习到，
和反向传播完全不同的权值调整算法

###	*Meta Reinforcement Learning*

MetaRL：RL智能体目标是最大化奖励，其通过不断提升自己的学习
算法来加速获取奖励，这也涉及到自我指涉

##	Additional Model

加法模型：将模型**视为**多个基模型加和而来

$$
f(x) = \sum_{m=1}^M \beta_m b(x;\theta_m)
$$

> - $b(x;\theta_m)$：基函数
> - $\theta_m$：基函数的参数
> - $\beta_m$：基函数的系数

-	则相应风险极小化策略

	$$
	\arg\min_{\beta_m, \theta_m} \sum_{i=1}^N
		L(y_i, \sum_{m=1}^M \beta_m b(x_i;\theta_m))
	$$

	> - $L(y, f(x))$：损失函数

###	Forward Stagewise Algorithm

前向分步算法：从前往后，每步只学习**加法模型**中一个基函数
及其系数，逐步逼近优化目标函数，简化优化复杂度

-	即每步只求解优化

	$$
	\arg\min_{\beta, \theta} \sum_{i=1}^N
		L(y_i, \hat f_m(x_i) + \beta b(x_i;\theta))
	$$

	> - $\hat f_m$：前m轮基函数预测值加和

####	步骤

> - 输入：训练数据集$T={(x_1,y_1), \cdots, (x_N,y_N)}$，损失
	函数$L(y,f(x))$，基函数集$\{b(x;\theta)\}$
> - 输出：加法模型$f(x)$

-	初始化$f_0(x)=0$

-	对$m=1,2,\cdots,M$，加法模型中M个基函数

	-	极小化损失函数得到参数$\beta_m, \theta_m$
		$$
		(\beta_m, \theta_m) = \arg\min_{\beta, \theta}
			\sum_{i=1}^N L(y_i, f_{m-1}(x_1) +
			\beta b(x_i; \theta))
		$$

	-	更新
		$$
		f_m(x) = f_{m-1}(x) + \beta_m b(x;y_M)
		$$

-	得到加法模型
	$$
	f(x) = f_M(x) = \sum_{i=1}^M \beta_m b(x;\theta_m)
	$$

###	AdaBoost&前向分步算法

AdaBoost（基分类器loss使用分类误差率）是前向分步算法的特例，
是由基本分类器组成的加法模型，损失函数是指数函数

-	基函数为基本分类器时加法模型等价于AdaBoost的最终分类器
	$f(x) = \sum_{m=1}^M \alpha_m G_m(x)$

-	前向分步算法的损失函数为指数函数$L(y,f(x))=exp(-yf(x))$
	时，学习的具体操作等价于AdaBoost算法具体操作

	-	假设经过m-1轮迭代，前向分步算法已经得到

		$$\begin{align*}
		f_{m-1}(x) & = f_{m-2}(x) + \alpha_{m-1}G_{m-1}(x) \\
			& = \alpha_1G_1(x) + \cdots +
			\alpha_{m-1}G_{m-1}(x)
		\end{align*}$$

	-	经过第m迭代得到$\alpha_m, G_m(x), f_m(x)$，其中

		$$\begin{align*}
		(\alpha_m, G_m(x)) & = \arg\min_{\alpha, G}
				\sum_{i=1}^N exp(-y_i(f_{m-1}(x_i) +
				\alpha G(x_i))) \\
			& = \arg\min_{\alpha, G} \sum_{i=1}^N \bar w_{m,i}
				exp(-y_i \alpha G(x_i))
		\end{align*}$$

		> - $\bar w_{m,i}=exp(-y_i f_{m-1}(x_i))$：不依赖
			$\alpha, G$

	-	$\forall \alpha > 0$，使得损失最小应该有
		（提出$\alpha$）

		$$\begin{align*}
		G_m^{*}(x) & = \arg\min_G \sum_{i=1}^N \bar w_{m,i}
				exp(-y_i f_{m-1}(x_i)) \\
			& = \arg\min_G \sum_{i=1}^N \bar w_{m,i}
				I(y_i \neq G(x_i))
		\end{align*}$$

		此分类器$G_m^{*}$即为使得第m轮加权训练误差最小分类器
		，即AdaBoost算法的基本分类器

	-	又根据

		$$\begin{align*}
		\sum_{i=1}^N \bar w_{m,i} exp(-y_i \alpha G(x_i)) & =
			\sum_{y_i = G_m(x_i)} \bar w_{m,i} e^{-\alpha} +
			\sum_{y_i \neq G_m(x_i)} \bar w_{m,i} e^\alpha \\
		& = (e^\alpha - e^{-\alpha}) \sum_{i=1}^N (\bar w_{m,i}
			I(y_i \neq G(x_i))) + e^{-\alpha}
			\sum_{i=1}^N \bar w_{m,i}
		\end{align*}$$

		带入$G_m^{*}$，对$\alpha$求导置0，求得极小值为

		$$\begin{align*}
		\alpha_m^{*} & = \frac 1 2 log \frac {1-e_m} {e_m} \\
		e_m & = \frac {\sum_{i=1}^N (\bar w_{m,i}
				I(y_i \neq G_m(x_i)))}
			{\sum_{i=1}^N \bar w_{m,i}} \\
		& = \frac {\sum_{i=1}^N (\bar w_{m,i}
				I(y_i \neq G_m(x_i)))} {Z_m} \\
		& = \sum_{i=1}^N w_{m,i} I(y_i \neq G_m(x_i))
		\end{align*}$$

		> - $w_{m,i}, Z_M$同AdaBoost中

		即为AdaBoost中$\alpha_m$

	-	对权值更新有

		$$
		\bar w_{m+1,i} = \bar w_{m,i} exp(-y_i \alpha_m G_m(x))
		$$

		与AdaBoost权值更新只相差规范化因子$Z_M$

