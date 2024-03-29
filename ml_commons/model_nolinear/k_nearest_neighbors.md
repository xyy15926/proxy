---
title: K-Nearest Neighor
categories:
  - ML Model
  - Nolinear Model
tags:
  - Machine Learning
  - ML Model
  - NoLinear Model
  - KNN
date: 2019-07-13 23:25:01
updated: 2021-07-16 16:05:22
toc: true
mathjax: true
comments: true
description: K-Nearest Neighor
---

##	K-NN

-	输入：p维实例特征向量
	-	将样本点视为p维特征空间的中点

-	输出：实例类别，可以取多类别

-	基本思想
	-	在已有数据中找到与$X_0$相似的若干个观测
		$(X_1, X_2, ..., X_k)$，称为$X_0$的近邻
	-	对近邻$(X_1, X_2, ..., X_k)$的输出变量
		$(y_1, y_2, ..., y_k)$，计算诸如算术平均值
		（加权平均值、中位数、众数），作为新观测$X_0$输出
		变量取值$y_0$的预测值

-	特点
	-	k近邻不具有显式学习过程、简单、直观
	-	不需要假设$y=f(X)$函数体形式，实际上是利用训练数据集
		对特征空间进行划分

###	局部方法

k-NN是一种“局部”方法，仅适合特征空间维度较低的情况

-	给定k的情况下，在高维空间中，需要到更远的区域寻找近邻，
	局部性逐渐丧失，近似误差变大

-	如：n个观测均匀分布在超立方体中，确定k后即确定$X_0$需要
	寻找的近邻个数占总观测的比率r，即近邻覆盖的体积

	-	考虑$X_0$在原点，则近邻分布的小立方体边期望长度为

		$$
		Ed_p(r) = r^{1/p} \\
		Ed_3(0.1) = 0.1^{1/3} = 0.46 \\
		Ed_10(0.1)d = 0.1^{1/10} = 0.79 \\
		Ed_10(0.01) = 0.1^{1/10} = 0.63 \\
		$$

	-	可以看出：减少近邻比例（数量）没有帮助，还会使得近似
		误差变大，只能通过增大样本量解决

-	特征选择有必要

###	特征选择

-	变量本身考察
	-	*low variance filter*：剔除标准差小于阈值数值型变量
	-	*missing values ratio*：剔除缺失值大于阈值的变量
	-	剔除众数比率大于阈值的分类型变量

-	变量与输出变量相关性角度考察
	-	*high correlation filter*

-	对预测误差影响角度考察
	-	Wrapper方法：逐个选择使错误率、均方误差下降最快变量
		，可使用*Forward Feature Elimination*

##	k-NN模型

K-NN是使用模型：实际上对应于特征空间的划分

-	模型包括3个基本要素，据此划分特征空间，确定特征空间中
	每个点所属类
	-	k值选择
	-	距离度量：参见*data_science/ref/functions*
	-	分类决策规则

###	k值选择

k值选择对k-NN方法有重大影响

-	较小k值：相当于使用较小邻域中训练实例进行预测

	-	复杂模型，容易发生过拟合
	-	*approximation error*较小：只有于输入实例近、相似的
		训练实例才会对预测结果有影响
	-	*estimation error*较大：预测结果对近邻实例点非常敏感

-	较大k值：相当于使用较大邻域中训练实例进行预测

	-	简单模型
	-	估计误差较小
	-	近似误差较大：同输如实例远、相似程度差的训练实例也会
		对预测结果有影响

####	k=1

只使用一个近邻做预测

-	找到距离$X_0$最近的近邻$X_i$，用其取值作为预测值

-	模型简单、效果较理想

	-	尤其适合特征空间维度较低、类别边界不规则情况
	-	只根据单个近邻预测，预测结果受近邻差异影响极大，预测
		波动（方差）大，稳健性低

-	预测错误的概率不高于普通贝叶斯方法的两倍

	$$\begin{align*}
	P_e & = (1-p(y=1|X=X_0))P(y=1|X=X_0) +
			(1-p(y=0|X=X_0))P(y=0|X=X_0) \\
		& = 2P(y=1|X=X_0)(1-P(y=1|X=X_0)) \\
		& \leq 2(1-P(y=1|X=X_0)) \\
	\end{align*}$$

	> - $P(y=1|X=X_0)$：普通贝叶斯方法做分类预测，预测结果
		为1的概率
	> - 1-NN方法犯错的概率就是$X_0$、$X_i$二者实际值不同的
		概率？？？？

####	k=N

使用训练样本整体做预测

-	无论输入实例，预测结果完全相同
	-	对分类预测，预测结果为“众数”
	-	对回归预测，预测结果为“平均数”

-	模型过于简单、效果不好
	-	忽略训练实例中大量信息
	-	“稳健性”极好：预测值基本不受近邻影响，无波动

###	决策规则

####	分类决策规则

#####	*Majority Voting Rule*

多数表决规则：等价于经验风险最小化

-	分类损失函数为0-1损失函数，分类函数为
	$f: \mathcal{R^n} \rightarrow \{c_1, c_2, \cdots\}$

-	误分类概率$P(Y \neq f(X)) = 1 - P(Y=f(X))$

-	给定实例$x \in \mathcal{X}$的误分率为

	$$
	\frac 1 k \sum_{x \in N_k(x)} I(y_i \neq c_j) = 
	1 - \frac 1 k \sum_{x \in N_k(x)} I(y_i = c_j)
	$$

	> - $N_k(x)$：最近邻k个实例构成集合
	> - $c_j$：涵盖$N_k(x)$区域的类别
	> - $I$：指示函数

-	为使误分率（经验风险）最小，应选择众数

> - 经验风险的构造中，前提是近邻被认为属于相同类别$c_j$，
> - 当然这个假设是合理的，因为k-NN方法就是认为近邻类别相同，
	并使用近邻信息预测
> - $c_j$的选择、选择方法是模型选择的一部分，不同的$c_j$会
	有不同的经验风险

###	数值决策规则

###	算法

-	实现k近邻法时，主要问题是对训练数据进行快速k近邻搜索，
	尤在特征空间维数大、训练数据量大

-	考虑使用特殊的结构存储训练数据，减少计算距离次数，提高
	k近邻搜索效率

####	*linear scan*

线性扫描：最简单的实现方法

-	需要计算输入实例与每个训练实例的距离，训练集较大时计算
	非常耗时

####	kd树最近邻搜索

> - 输入：已构造kd树
> - 输出：x的最近邻

-	在kd树种找出包含目标点x的叶节点的

	-	从根节点出发，比较对应坐标，递归进行访问，直到叶节点
		为止

	-	目标点在训练样本中不存在，必然能够访问到叶节点

-	以此叶节点为“当前最近点”

	-	目标点在此叶节点中点所在的区域内，且区域内只有该
		叶节点中点

-	回溯，并在每个节点上检查

	-	如果当前节点保存实例点比当前最近点距离目标的更近，
		更新该实例点为“当前最近点”

	-	检查该节点另一子区域是否可能具有更近距离的点

		-	即其是否同以目标点为圆心、当前最短距离为半径圆
			相交
		-	只需要比较目标点和相应坐标轴距离和最短距离即可

	-	若二者相交，则将目标节点视为**属于**该子区域中点，
		进行最近邻搜索，**递归向下**查找到相应叶节点，重新
		开始回退

	-	若二者不相交，则继续回退

-	退回到根节点时，搜索结束，最后“当前最近点”即为最近邻点

> - 这里涉及到回溯过程中，另一侧子域是否访问过问题，可以通过
	标记、比较相应轴坐标等方式判断
> - k>1的情况类似，不过检测时使用最远近邻，新近邻需要和所有
	原近邻依次比较

##	加权k-NN

###	变量重要性

计算变量的加权距离，重要变量赋予较高权重

-	变量重要性：*Backward Feature Elimination*得到各变量
	重要性排序

	$$
	FI_{(i)} = e_i + \frac {1} {p} \quad \\
	w_{(i)} = \frac {FI_{(i)}} {\sum_{j=1}^p FI_{(j)}}
	$$

	> - $e_i$：剔除变量i之后的均方误差（错误率）

-	加权距离：$d_w(x,y)=\sqrt {\sum_{i=1}^{p} w^{(i)}(x_i - y_i)^2}$


###	观测相似性

目标点的k个近邻对预测结果不应有“同等力度”的影响，与$X_0$越
相似的观测，预测时重要性（权重）越大

-	权重：用函数$K(d)$将距离d转换相似性，$K(d)$应该有特性

	> - 非负：$K(d) \geqslant 0, d \in R^n$
	> - 0处取极大：$max K(d) = K(0)$
	> - 单调减函数，距离越远，相似性越小

	-	核函数符合上述特征
	-	且研究表明除均匀核外，其他核函数预测误差差异均不明显

####	步骤

-	依据函数距离函数$d(Z_{(i)}, Z_0)$找到$X_0$的k+1个近邻

	-	使用第k+1个近邻距离作为最大值，调整距离在0-1之间
		$$
		D(Z_{(i)}, Z_0) = \frac {d(Z_{(i)}, Z_0)}
			{d(Z_{(k+1)}, Z_0)}, \quad i=1,2,...,k
		$$

-	依据函数$w_i=K(d)$确定k各近邻的权重

-	预测
	-	回归预测
		$$\hat{y}_0 = \frac 1 k (\sum_{i=1}^k w_iy_i)$$
	-	分类预测：多数表决原则
		$$
		\hat{y}_0 = max_r (\sum_{i=1}^k w_iI(y_i=r)) \\
		P(\hat{y}_0=r|X_0)= \frac
			{\sum_{i=1}^k w_iI(y_i=r)} {\sum_{i=1}^k w_i}
		$$

##	*Approximate Nearest Neighbor*

相似最近邻

