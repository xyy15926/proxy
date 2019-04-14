#	Shallowing Learning

浅层学习：传统机器学习

##	*Supervised Learning*

监督学习：学习一个模型，使得模型能够对任意给定输入、输出，
做出好的预测

-	从给定的、有限的、用于学习的*train data*
	$T=\{(x_1,y_1), (x_2,y_2), \cdots, (x_N, y_N)\}$中学习
-	预测“未知”*test data*
	$T=\{(x_1,y_1), (x_2,y_2), \cdots, (x_N^{'}, y_N^{'})\}$

###	数据

> - *input space*：输入空间$\chi$，所有输入$X$可能取值的集合

> - *output space*：输出空间$\gamma$，所有输出$Y$可能取值
	集合

> - *feature space*：特征空间，表示输入实例*feature vector*
	存在的空间

> > -	特征空间每维对应一个特征
> > -	模型实际上是定义在特征空间上的
> > -	有时特征空间等于输入空间，有时特征空间时输入空间的
		象集

###	学习方法分类

####	*Generative Approach*

生成方法：由数据学习联合概率分布$P(X, Y)$，然后求出条件概率
分布$P(Y|X)$作为*generative model*

-	方法学习给定输入X产生输出Y的生成关系（联合概率分布）

-	*generative model*：生成模型，由生成方法学习到的模型
	$P(Y|X) = \frac {P(X, Y)} {P(X}$

	-	朴素贝叶斯法
	-	隐马尔可夫模型

-	特点

	-	可以还原联合概率分布$P(X, Y)$

	-	生成方法学习收敛速度快，样本容量增加时，学习到的模型
		可以快速收敛到真实模型

	-	存在隐变量时，仍可以使用生成方法学习

####	*Discriminative Approach*

判别方法：由数据直接学习决策函数$f(x)$、条件概率分布$P(Y|X)$
作为*discriminative model*

-	判别方法关心的是对给定输入X，预测输出Y

-	*discriminative model*：判别模型

	-	k近邻
	-	感知机
	-	决策树
	-	逻辑回归
	-	最大熵模型
	-	支持向量机
	-	提升方法
	-	条件随机场

-	特点

	-	直接学习条件概率、决策函数
	-	直面预测，学习准确率更高
	-	可以对数据进行各种程度抽象、定义特征、使用特征，简化
		学习问题

###	问题分类

-	*well-posed problem*：好解问题，指问题解应该满足以下条件

	-	解存在
	-	解唯一
	-	解行为随着初值**连续变化**

-	*ill-posed problem*：病态问题，解不满足以上三个条件

####	*Classification*

分类问题：输出变量$Y$为有限个离散变量

-	学习过程：根据已知训练数据集，利用有效学习方法学习分类器
	$P(Y|X))$、$Y=F(X)$

-	分类过程：利用学习的分类器对新输入实例进行分类

-	可用学习方法

	-	k近邻
	-	感知机
	-	朴素贝叶斯
	-	决策树
	-	决策列表
	-	逻辑回归
	-	支持向量机
	-	提升方法
	-	贝叶斯网络
	-	神经网络
	-	winnow

-	不存在分类能力弱于随机预测的分类器（结论取反）

####	*Tagging*

标注问题：输入、输出**均为变量序列**

-	可认为是分类问题的一个推广、更复杂*structure prediction*
	简单形式

-	学习过程：利用已知训练数据集构建条件概率分布模型
	$P(Y^{(1)}, Y^{(2)}, \cdots, Y^{(n)}|X^{(1)}, X^{(2)}, \cdots, X^{(n)})$

	> - $X^{(1)}, X^{(2)}, \cdots, X^{(n)}$：每个输入序列
	> - $Y^{(1)}, Y^{(2)}, \cdots, Y^{(n)}$：所有可能标记

-	标注过程：按照学习到的条件概率分布，标记新的输入观测序列

-	可用模型

	-	隐马尔可夫模型
	-	条件随机场

####	*Regression*

回归问题：输入（自变量）、输出（因变量）均为连续变量

-	回归模型的拟合等价于函数拟合：选择函数曲线很好的拟合已知
	数据，且很好的预测未知数据

-	学习过程：基于训练数据构架模型（函数）$Y=f(X)$

	-	最常用损失函数是平方损失函数，此时可以使用最小二乘
		求解

-	预测过程：根据学习到函数模型确定相应输出

##	*Unsupervised Learning*

无监督学习：没有给定实现标记过的训练示例，自动对输入的数据
进行分类

-	主要目标：预训练一般模型（称识别、编码）网络，供其他任务
	使用

-	目前为止，监督模型总是比无监督的预训练模型表现得好，主要
	原因是监督模型对数据的**特性编码**更好

###	问题分类

####	*Clustering*

聚类

-	*Hierarchy Clustering*
-	*K-means*
-	*Mixture Models*
-	*DBSCAN*
-	*OPTICS Algorithm*

####	*Anomaly Detection*

异常检测

-	*Local Outlier Factor*

####	*Neural Networks*

神经网络

-	*Auto-encoders*
-	*Deep Belief Nets*
-	*Hebbian Learning*
-	*Generative Adversarial Networks*
-	*Self-organizing Map*

####	隐变量学习

-	*Expectation-maximization Algorithm*
-	*Methods of Moments*
-	*bind signal separation techniques*
	-	*Principal Component analysis*
	-	*Independent Component analysis*
	-	*Non-negative matrix factorization*
	-	*Singular Value Decomposition*

##	*Semi-Supervised Learning*

半监督学习

##	*Reinforcement Learning*

强化学习

##	学习要素

###	*Model*/*Hypothesis*/*Opimizee*/*Learner*/*Learning Algorithm*

模型/假说/优化对象/学习器/学习算法：模型就是要学习的条件概率
分布$P(Y|X)$、决策函数$Y=f(X)$

-	概率模型：用条件概率分布$P(Y|X)$表示的模型
-	非概率模型：用决策函数$Y=f(x)$表示的模型

> - *learner*：某类模型的总称
> - *hypothesis*：训练好的模型实例，有时也被强调作为学习器
	应用在某个样本集（如训练集）上得到的结果
> - *learning algorithm*：模型、策略、算法三者的模型总体

####	*Hypothesis Space*

假设空间：特征空间（输入空间）到输出空间的映射集合

-	假设空间可以定义为决策函数/条件概率的集合，通常是由参数
	向量$\theta$决定的函数/条件分布族

	-	假设空间包含所有可能的条件概率分布或决策函数
	-	假设空间的确定意味着学习范围的确定

-	概率模型假设空间可表示为：
	$F=\{P|P_{\theta}(Y|X), \theta \in R^n\}$

-	非概率模型假设空间可表示为：
	$F=\{f|Y=f(x),\Theta \in R^n \}$

> - 以下大部分情况使用决策函数，同时也可以代表概率分布

###	*Strategy*/*Goal*

策略/目标：从假设空间中，根据*evaluation criterion*选择最优
模型，使得其对已知训练数据、未知训练数据在给定评价准则下有
最优预测

-	选择合适策略，监督学习问题变为经验风险、结构风险函数
	**最优化问题**

-	在某些学习方法中，最优化问题目标函数也有可能不是风险函数
	，如：SVM，是和模型紧密相关的损失函数，但逻辑是一样的

####	*Empirical Risk Minimiation*

*ERM*：经验风险最小化策略认为，经验风险最小模型就是最优模型

-	按经验风险最小化求最优模型，等价于求最优化问题

	$$
	\min_{f \in F} \frac 1 N \sum_{i=1}^N L(y_i, f(x_i))
	$$

-	样本容量足够大时，经验风险最小化能保证有较好的学习效果，
	现实中也被广泛采用

####	*Structural Risk Minimization*

*SRM*：结构风险最小化，为防止过拟合提出的策略

-	结构化风险最小化策略认为结构风险最小的模型是最优模型，
	则求解最优模型等价于求解最优化问题

	$$
	arg \min_{f \in F} \frac 1 N \sum_{i=1}^N L(y_i, f(x_i))
		+ \lambda J(f)
	$$

-	结构风险小需要经验风险与模型复杂度同时小，此时模型往往
	对训练数据、未知的测试数据都有较好的预测

-	结构化风险最小策略符合*Occam's Razor*原理

> - *Occam's Razor*：奥卡姆剃刀原理，在所有可能选择的模型中
	，能够很好的解释已知数据并且十分简单才是最好的模型

###	*Algorithm*/*Optimizer*

算法/优化器：学习模型（选择、求解最优模型）的具体计算方法
（求解最优化问题）

-	如果最优化问题有显式解析解，比较简单

-	但通常解析解不存在，需要用数值计算方法求解
	-	保证找到全局最优解
	-	高效求解

