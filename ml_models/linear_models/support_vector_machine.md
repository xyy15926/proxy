---
title: Support Vector Machine
tags:
  - Model
  - Linear Model
  - SVM
categories:
  - ML Model
  - Linear Model
date: 2019-07-13 23:23:47
updated: 2019-07-13 12:03:11
toc: true
mathjax: true
comments: true
description: Support Vector Machine
---

##	总述

支持向量机是二分类模型

###	学习要素

-	基本模型：定义在特征空间上的间隔最大线性分类器

-	学习策略：间隔最大化

	-	可形式化为求解凸二次规划问题，也等价于正则化的合页
		损失函数最小化问题

	-	间隔最大使其有别于感知机，训练数据线性可分时分离
		超平面唯一

		-	误分类最小策略（0-1损失）得到分离超平面的解无穷
			多个
		-	距离和最小策略（平方损失）得到分离超平面唯一，
			但与此不同

-	学习算法：求解凸二次规划的最优化算法

###	数据

在给定特征空间上的训练数据集
$T=\{(x_1,y_1), (x_2,y_2),\cdots,(x_N,y_N)\}$，其中
$x_i \in \mathcal{X} = R^n, y_i \in {-1, +1}, i=1,2,...,N$

> - 输入空间：欧氏空间或离散集合
> - 特征空间：希尔伯特空间
> - 输出空间：离散集合

-	SVM假设输入空间、特征空间为两个不同空间，SVM在特征空间
	上进行学习

-	线性（可分）支持向量机假设两个空间元素**一一对应**，并将
	输入空间中的输入映射为特征空间中特征向量

-	非线性支持向量机利用，输入空间到特征空间的非线性映射
	（核函数）将输入映射为特征向量

###	概念

##	*Linear Support Vector Machine in Linearly Separable Case*

线性可分支持向量机：硬间隔支持向量机，训练数据线性可分时，
通过硬间隔最大化策略学习

-	直观含义：不仅将正负实例分开，而且对于最难分的实例点
	（离超平面最近的点），也有足够大的确信度将其分开，这样的
	超平面对新实例预测能力较好

> - *Hard-margin Maximization*：硬间隔最大化，最大化超平面
	$(w,b)$关于线性可分训练数据集的两类样本集几何间隔

###	原始问题策略

-	约束最优化问题表述

	$$\begin{align}
	\max_{w,b} & \gamma & \\
	s.t. & \frac {y_i} {\|w\|} (wx + b) \geq \gamma,
		i=1,2,\cdots,N
	\end{align} $$

-	考虑函数间隔、几何间隔关系得到问题
	-	目标、约束中使用函数间隔表示几何间隔
	-	就是普通**等比缩放**、**分母移位**，不用考虑太多

	$$\begin{align*}
	\max_{w,b} & \frac {\hat{\gamma}} {\|w\|} \\
	s.t. & y_i(wx_i + b) \geq \hat{\gamma}, i=1,2,\cdots,N
	\end{align*}$$

-	而函数间隔$\hat{\gamma}$大小会随着超平面参数变化
	成比例变化，其取值对问题求解无影响，所以可取
	$\hat{\gamma}=1$带入，得到最优化问题

	$$\begin{align*}
	\min_{w,b} & \frac 1 2 {\|w\|}^2 \\
	s.t. & y_i(wx_i + b) - 1 \geq 0, i=1,2,\cdots,N
	\end{align*}$$

	> - $\max \frac 1 {\|w\|}$和
		$\min \frac 1 2 {\|w\|}^2$等价

	-	这里确实没有通用变换技巧，因为这里的$\hat \gamma$
		是特殊的值，其取值与$w,b$相关，这是**问题自然蕴含**
		，可以视为还有以下一个依赖

	-	当然也可以直接证明两个问题等价：先证明最优解在等式
		成立时取到，然后目标函数中1替换为等式左边

###	最大间隔分离平面存在性

> - 若训练数据集T线性可分，则可将训练数据集中样本点完全
	正确分开的最大间隔分离超平面存在且唯一

####	存在性

-	训练数据集线性可分，所以以上中最优化问题一定存在可行解
-	又目标函数又下界，则最优化问题必有解
#todo
-	又训练数据集中正、负类点都有，所以$(0,b)$必不是最优化
	可行解

####	唯一性

-	若以上最优化问题存在两个最优解$(w_1^{*},b_1^{*})$、
	$w_2^{*},b_2^{*}$
	
-	显然$\|w_1^{*}\| = \|w_2^{*}\| = c$，
	$(w=\frac {w_1^{*}+w_2^{*}} 2,b=\frac {b_1^{*}+b_2^{*}} 2)$
	使最优化问题的一个可行解，则有

	$$
	c \leq \|w\| \leq \frac 1 2 \|w_1^{*}\| + \frac 1 2
		\|w_2^{*} = c
	$$

-	则有$\|w\|=\frac 1 2 \|w_1^{*}\|+\frac 1 2 \|w_2^{*}\|$
	，有$w_1^{*} = \lambda w_2^{*}, |\lambda|=1$

	-	$\lambda = -1$，不是可行解，矛盾
	-	$\lambda = 1$，则$w_1^{(*)} = w_2^{*}$，两个最优解
		写为$(w^{*}, b_1^{*})$、$(w^{*}, b_2^{*})$

-	设$x_1^{+}, x_1^{-}, x_2^{+}, x_2^{-}$分别为对应以上两组
	超平面，使得约束取等号、正/负类别的样本点，则有

	$$\begin{align*}
	b_1^{*} & = -\frac 1 2 (w^{*} x_1^{+} + w^{*} x_1^{-}) \\
	b_2^{*} & = -\frac 1 2 (w^{*} x_2^{+} + w^{*} x_2^{-}) \\
	\end{align*}$$

	则有

	$$
	b_1^{*} - b_2^{*} = -\frac 1 2 [w^{*}(x_1^{+} - x_2^{+})
		+ w^{*} (x_1^{-} - x_2^{-})]
	$$

-	又因为以上支持向量的性质可得

	$$\begin{align*}
	w^{*}x_2^{+} + b_1^{*} & \geq 1 = w^{*}x_1^{+} + b_1^{*} \\
	w^{*}x_1^{+} + b_2^{*} & \geq 1 = w^{*}x_2^{+} + b_2^{*}
	\end{align*}$$

	则有$w^{*}(x_1^{+} - x_2^{+})=0$，同理有
	$w^{*}(x_1^{-} - x_2^{-})=0$

-	则$b_1^{*} - b_2^{*} = 0$

###	概念

####	*Support Vector*

支持向量：训练数据集中与超平面距离最近的样本点实例

-	在线性可分支持向量机中即为使得约束条件取等号成立的点
-	在决定分离超平面时，只有支持向量起作用，其他实例点不起
	作用
-	支持向量一般很少，所以支持向量机由很少的“重要”训练样本
	决定

####	间隔边界

间隔边界：超平面$wx + b = +/-1$

-	支持向量位于其上
-	两个间隔边界之间距离称为间隔$=\frac 2 {\|w\|}$

###	算法

> - 输入：线性可分训练数据集T
> - 输出：最大间隔分离超平面、分类决策函数

1.	构造并求解约束最优化问题

	$$\begin{align*}
	\min_{w,b} &  \frac 1 2 {\|w\|}^2 \\
	s.t. & y_i(wx_i + b) - 1 \geq 0, i=1,2,\cdots,N
	\end{align*}$$

	得到最优解$w^{*}, b^{*}$

2.	得到分离超平面

	$$
	w^{*}x + b^{*} = 0
	$$

	分类决策函数

	$$
	f(x) = sign(w^{*}x + b^{*})
	$$

###	多分类

-	*1 vs n-1*：对类$k=1,...,n$分别训练当前类对剩余类分类器
	-	分类器数据量有偏，可以在负类样本中进行抽样
	-	训练n个分类器

-	*1 vs 1*：对$k=1,...,n$类别两两训练分类器，预测时取各
	分类器投票多数
	-	需要训练$\frac {n(n-1)} 2$给分类器

-	*DAG*：对$k=1,...,n$类别两两训练分类器，根据预先设计的、
	可以使用DAG表示的分类器预测顺序依次预测
	-	即排除法排除较为不可能类别
	-	一旦某次预测失误，之后分类器无法弥补
		-	但是错误率可控
		-	设计DAG时可以每次选择相差最大的类别优先判别

##	*Dual Algorithm*

对偶算法：求解对偶问题得到原始问题的最优解

-	对偶问题往往更容易求解
-	自然引入核函数，进而推广到非线性分类问题

###	对偶问题策略

-	Lagrange函数如下

	$$
	L(w,b,\alpha) = \frac 1 2 \|w\|^2 - \sum_{i=1}^N
		\alpha_i y_i (wx_i + b) + \sum_{i=1}^N \alpha_i
	$$

	> - $\alpha_i > 0$：*Lagrange multiplier*

-	根据拉格朗日对偶性，原始问题的对偶问题是极大极小问题

	$$
	\max_{\alpha} \min_{w,b} L(w,b,\alpha)
	$$


-	求$\min_{w,b} L(w,b,\alpha)$，对拉格朗日函数求偏导置0

	$$\begin{align*}
	\triangledown_w L(w,b,\alpha) & = w - \sum_{i=1}^N
		\alpha_i y_i x_i = 0 \\
	\triangledown_b L(w,b,\alpha) & = -\sum_{i=1}^N
		\alpha_i y_i = 0
	\end{align*}$$

	解得

	$$\begin{align*}
	w = \sum_{i=1}^N \alpha_i y_i x_i \\
	\sum_{i=1}^N \alpha_i y_i = 0
	\end{align*}$$

-	将以上结果代理拉格朗日函数可得

	$$\begin{align*}
	L(w,b,\alpha) & = \frac 1 2 \sum_{i=1}^N \sum_{j=1}^N
		\alpha_i \alpha_j y_i y_j (x_i x_j) - \sum_{i=1}^N
		\alpha_i y_i ((\sum_{j=1}^N \alpha_j y_j x_j)x_i + b)
		+ \sum_{i=1}^N \alpha_i \\
	& = -\frac 1 2 \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j
		y_i y_i (x_i x_j) + \sum_{i=1}^N \alpha_i
	\end{align*}
	$$

-	以上函数对$\alpha$极大即为对偶问题，为方便改成极小

	$$\begin{align*}
	\min_{\alpha} & \frac 1 2 \sum_{i=1}^N \sum_{j=1}^N
		\alpha_i \alpha_j y_i y_j (x_i x_j) -
		\sum_{i=1}^N \alpha_i \\
	s.t. & \sum_{i=1}^N \alpha_i y_i = 0, \\
	& \alpha_i > 0, i = 1,2,\cdots,N
	\end{align*}$$

###	原始问题解

> - 设$\alpha^{*} = (\alpha_1^{*}, \cdots, \alpha_N^{*})$是
	上述对偶优化问题的解，则存在$j \in [1, N]$使得
	$\alpha_j^{*} > 0$，且原始最优化问题解如下
	$$\begin{align*}
	w^{*} & = \sum_{i=1}^N \alpha_i^{*} y_i x_i \\
	b^{*} & = y_j - \sum_{i=1}^N \alpha_i^{*} y_i (x_i x_j)
	\end{align*}$$

-	由KKT条件成立有

	$$\begin{align*}
	& \triangledown_w L(w^{*}, b^{*}, \alpha^{*}) = w^{*} -
		\sum_{i=1}^N \alpha_i{*} y_i x_i = 0 \\
	& \triangledown_b L(w^{*}, b^{*}, \alpha^{*}) =
		-\sum_{i=1}^N \alpha_i^{*} y_i = 0 \\
	& \alpha^{*}(y_i (w^{*} x_i + b^{*}) - 1) = 0,
		i = 1,2,\cdots,N \\
	& y_i(w^{*} x_i + b) - 1 \geq 0, i = 1,2,\cdots,N \\
	& \alpha_i^{*} \geq 0, i = 1,2,\cdots,N \\
	\end{align*}
	$$

-	可得

	$$
	w^{*} = \sum_i \alpha_i^{*} y_i x_i
	$$

-	其中至少有一个$\alpha_j > 0$，否则$w^{*}=0$不是原问题解
	，且有

	$$\begin{align*}
	y_j(w^{*} x_j + b^{*}) - 1 = 0
	\end{align*}$$

	注意到$y_j \in \{-1, +1\}$，则有

	$$
	b^{*} = y_j - \sum_{i=1}^N \alpha_i^{*} y_i (x_i x_j)
	$$

####	分离超平面

-	则分离超平面为
	$$
	\sum_{i=1}^N \alpha_i^{*} y_i (x x_i) + b^{*} = 0
	$$

-	分类决策函数为
	$$
	f(x) = sgn(\sum_{i=1}^N \alpha_i^{*} y_i (x x_i) + b^{*})
	$$
	即分类决策函数只依赖输入$x$和训练样本的内积

####	支持向量

> - 将对偶最优化问题中，训练数据集中对应$\alpha_i^{*} > 0$
	的样本点$(x_i, y_i)$称为支持向量

-	由KKT互补条件可知，对应$\alpha_i^{*} > 0$的实例$x_i$有
	$$
	y_i(w^{*} x_i + b^{*}) - 1 = 0
	$$
	即$(x_i, y_i)$在间隔边界上，同原始问题中支持向量定义一致

###	算法

> - 输入：线性可分数据集$T$
> - 输出：分离超平面、分类决策函数

1.	构造并求解以上对偶约束最优化问题，求得最优解
	$\alpha^{*} = (\alpha_1^{*},...,\alpha_N^{*})^T$

2.	依据以上公式求$w^{*}$，选取$\alpha^{*}$正分量
	$\alpha_j^{*} > 0$计算$b^{*}$

3.	求出分类超平面、分类决策函数

##	*Linear Support Vector Machine*

线性支持向量机：训练数据线性不可分时，通过软间隔最大化策略
学习

-	训练集线性不可分通常是由于存在一些*outlier*

	-	这些特异点不能满足函数间隔大于等于1的约束条件
	-	将这些特异点除去后，剩下大部分样本点组成的集合是
		线性可分的

-	对每个样本点$(x_i,y_i)$引入松弛变量$\xi_i \geq 0$

	-	使得函数间隔加上松弛变量大于等于1
	-	对每个松弛变量$\xi_i$，支付一个代价$\xi_i$

> - *soft-margin maximization*：软间隔最大化，最大化样本点
	几何间隔时，尽量减少误分类点数量

###	策略

线性不可分的线性支持向量机的学习变成如下凸二次规划，即
软间隔最大化

$$\begin{align*}
\min_{w,b,\xi} & \frac 1 2 \|w\|^2 + C \sum_{i=1}^N \xi_i \\
s.t. & y_i(w x_i + b) \geq 1 - \xi_i, i=1,2,\cdots,N \\
& \xi_i \geq 0, i=1,2,\cdots,N
\end{align*}
$$

> - $\xi_i$：松弛变量
> - $C > 0$：惩罚参数，由应用问题决定，C越大对误分类惩罚越大

-	最小化目标函数包含两层含义

	-	$\frac 1 2 \|w\|^2$尽量小，间隔尽量大
	-	误分类点个数尽量小

-	以上问题是凸二次规划问题，所以$(w,b,\xi)$的解是存在的

	-	$w$解唯一
	-	$b$解可能不唯一，存在一个区间

> - 对给定的线性不可分训练数据集，通过求解以上凸二次规划问题
	得到的分类超平面
	$$ w^{*} x + b^{*} = 0 $$
	以及相应的分类决策函数
	$$ f(x) = sgn(w^{*} x + b^{*}) $$
	称为线性支持向量机

-	线性支持向量包括线性可分支持向量机
-	现实中训练数据集往往线性不可分，线性支持向量机适用性更广

###	对偶问题策略

原始问题的对偶问题

$$
\begin{align*}
\min_\alpha & \frac 1 2 \sum_{i=1}^N \sum_{j=1}^N \alpha_i
	\alpha_j y_i y_j (x_i x_j) - \sum_{i=1}^N \alpha_i \\
s.t. & \sum_{i=1}^N \alpha_i y_i = 0 \\
& 0 \leq \alpha_i \leq C, i=1,2,\cdots,N
\end{align*}
$$

-	类似线性可分支持向量机利用Lagrange对偶即可得

###	原始问题解

> - 设$\alpha^{*} = (\alpha_1^{*}, \cdots, \alpha_N^{*})$是
	上述对偶优化问题的解，则存在$j \in [1, N]$使得
	$0 < \alpha_j^{*} < C$，且原始最优化问题解如下
	$$\begin{align*}
	w^{*} & = \sum_{i=1}^N \alpha_i^{*} y_i x_i \\
	b^{*} & = y_j - \sum_{i=1}^N \alpha_i^{*} y_i (x_i x_j)
	\end{align*}$$

-	类似线性可分支持向量机利用KKT条件即可得

####	分离超平面

-	则分离超平面为
	$$
	\sum_{i=1}^N \alpha_i^{*} y_i (x x_i) + b^{*} = 0
	$$

-	分类决策函数/线性支持向量机对偶形式
	$$
	f(x) = sgn(\sum_{i=1}^N \alpha_i^{*} y_i (x x_i) + b^{*})
	$$
	即分类决策函数只依赖输入$x$和训练样本的内积

####	支持向量

> - 将对偶最优化问题中，训练数据集中对应$\alpha_i^{*} > 0$
	的样本点$x_i$称为（软间隔）支持向量

-	$\alpha_i^{*} < C$：则$\xi_i = 0$，恰好落在间隔边界上
-	$\alpha_i^{*} = C, 0 < \xi_i < 1$：间隔边界与分离超平面
	之间
-	$\alpha_i^{*} = C, \xi=1$：分离超平面上
-	$\alpha_i^{*} = C, \xi>1$：分离超平面误分一侧

###	*Hinge Loss*

线性支持向量机策略还可以视为最小化以下目标函数

$$
\sum_{i=1}^N [1-y_i(w x_i + b)]_{+} + \lambda \|w\|^2
$$

> - 第一项：经验风险，合页损失函数
> - 第二项：正则化项

-	$w$模越大，间隔越小，合页损失越小，所以用这个作为
	正则化项是合理的

> - 参见*data_science/loss*

####	等价证明

令

$$
\xi_i = [1 - y_i(w x_i + b)]_{+}
$$

-	则有$\xi_i \geq 0$

-	且

	$$\left \{ \begin{align*}
	& y_i(w x_i + b) = 1 - \xi_i, & y_i(w x_i + b) \leq 1 \\
	& y_i(w x_i + b) > 1 - \xi_i = 1, & y_i(w x_i + b) > 1
	\end{align*} \right.$$

	所以有

	$$
	y_i(w x_i + b) \geq 1 - \xi_i
	$$

-	则原问题两个约束条件均得到满足，此问题可写成

	$$
	\min_{w,b} \sum_{i=1}^N \xi_i + \lambda \|w\|^2
	$$

	取$\lambda = \frac 1 {2C}$，即同原问题

##	*Non-Linear Support Vector Machine*

非线性支持向量机

-	非线性问题：通过非线性模型才能很好进行分类的问题
	-	通过**非线性变换**$\phi(x)$，将输入空间映射到特征
		空间（维数可能非常高）
	-	原空间中非线性可分问题在特征空间可能变得线性可分，
		在特征空间中学习分类模型

-	SVM求解非线性问题时
	-	*kernel trick*：通过非线性变换将输入空间对应一个特征
		空间，使得输入空间中超曲面对应于特征空间的超平面模型
	-	软间隔最大化策略：在特征空间中求解线性支持向量机

###	*Kernal Trick*

核技巧：线性SVM的对偶问题中，目标函数、决策函数均只涉及输入
实例、实例之间的内积

-	将内积使用核函数代替，等价进行复杂的非线性变换

-	映射函数是非线性函数时，学习的含有核函数的支持向量机
	是非线性模型

-	-	学习是隐式地在特征空间中进行的，不需要显式定义特征
	空间、映射函数

> - 参见*data_science/ref/functions*

####	对偶问题目标函数

$$
W(\alpha) = \frac 1 2 \sum_{i=1}^N \sum_{j=1}^N
	\alpha_i \alpha_j y_i y_j K(x_i, x_j) -
	\sum_{i=1}^N \alpha_i
$$

###	原始问题解

####	分类决策函数

$$\begin{align*}
f(x) & = sgn(\sum_{i=1}^{N_s} \alpha_i^{*} y_i \phi(x_i)
	\phi(x) + b^{*}) \\
& = sgn(\sum_{i=1}^{N_s} \alpha_i^{*} y_i K(x_i, x) + b^{*})
\end{align*}$$

###	算法

> - 输入：训练数据集$T$
> - 输出：分类决策函数

1.	选取适当的核函数$K(x,z)$、适当参数C，构造求解最优化问题

	$$\begin{align*}
	\min_\alpha & \frac 1 2 \sum_{i=1}^N \sum_{j=1}^N
		\alpha_1 \alpha_j y_i y_j K(x_i, x_j) -
		\sum_{i=1}^N \alpha_i \\
	s.t. & \sum_{i=1}^N \alpha_i y_i = 0 \\
	& 0 \leq \alpha_i \leq C, i=1,2,...,N
	\end{align*}
	$$

	求得最优解
	$\alpha^{*} = (\alpha_1^{*},\alpha_2^{*},\cdots,\alpha_N^{*})$

2.	选择$\alpha^{*}$的一个正分量$0 < \alpha_j^{*} < C$，计算

	$$
	b^{*} = y_j - \sum_{i=1}^N \alpha_i^{*} y_i K(x_i x_j)
	$$

3.	构造决策函数

> - $K(x,z)$是正定核函数时，最优化问题是凸二次规划，有解

##	*Sequential Minimal Optimization*

序列最小最优化算法，主要解如下凸二次规划的对偶问题

$$\begin{align*}
\min_\alpha & \frac 1 2 \sum_{i=1}^N \sum_{j=1}^N
	\alpha_1 \alpha_j y_i y_j K(x_i, x_j) -
	\sum_{i=1}^N \alpha_i \\
s.t. & \sum_{i=1}^N \alpha_i y_i = 0 \\
& 0 \leq \alpha_i \leq C, i=1,2,...,N
\end{align*}$$

> - 凸二次规划有很多算法可以求得全局最优解，但是在训练样本
	容量较大时，算法会很低效

###	思想

将原问题不断分解为子问题求解，进而求解原问题

-	如果所有变量的解都满足此优化问题的KKT条件，得到此最优化
	问题的一个可能解
	
	-	对凸二次规划就是最优解，因为凸二次规划只有一个稳定点

-	否则选择两个变量，固定其他变量，构建一个二次规划

	-	目标是使得解符合KKT条件，但是因为等式约束的存在，
		不可能**单独改变一个变量**而保持等式约束

	-	子问题有解析解，求解速度快

	-	二次规划问题关于两个变量的解会使得原始二次规划的目标
		函数值变得更小，更接近原始二次规划的解
		（这里SMO原始论文有证明，违反KKT条件的变量可以做到）

-	不失一般性，假设选择两个变量$\alpha_1, \alpha_2$，其他
	变量$\alpha_i(i=3,4,...,N)$是固定的，则SMO最优化子问题

	$$\begin{align*}
	\min_{\alpha_1, \alpha_2} & W(\alpha_1, \alpha_2) =
		\frac 1 2 K_{11} \alpha_1^2 + \frac 1 2
		K_{22} \alpha_2^2 + y_1 y_2 K_{12} \alpha_1 \alpha_2
		- (\alpha_1 + \alpha_2) + y_1 \alpha_1 \sum_{i=3}^N
		y_i \alpha_i K_{i1} + y_2 \alpha_2 \sum_{i=1}^N
		y_i \alpha_i K{i2} \\
	s.t. & \alpha_1 y_1 + \alpha_2 y_2 = -\sum_{i=3}^N
		y_i \alpha_i = \zeta \\
	& 0 \leq \alpha_i \leq C, i=1,2
	\end{align*}$$

	> - $K_{ij} = K(x_i, x_j), i,j=1,2,\cdots,N$
	> - $\zeta$：常数

###	两变量二次规划取值范围

-	由**等式约束**，$\alpha_1, \alpha_2$中仅一个自由变量，
	不妨设为$\alpha_2$

	-	设初始可行解为$\alpha_1, \alpha_2$

	-	设最优解为$\alpha_1^{*}, \alpha_2^{*}$

	-	未经剪辑（不考虑约束条件而来得取值范围）最优解为
		$\alpha_2^{**}$

-	由**不等式**约束，可以得到$\alpha_2$取值范围$[L, H]$

	-	$y_1 = y_2 = +/-1$时

		$$\begin{align*}
		H & = \min \{C, \alpha_1 + \alpha_2 \} \\
		L & = \max \{0, \alpha_2 + \alpha_1 - C \}
		\end{align*}$$

	-	$y_1 \neq y_2$时

		$$\begin{align*}
		L & = \max \{0, \alpha_2 - \alpha_1 \} \\
		L & = \min \{C, C + \alpha_2 - \alpha_1 \}
		\end{align*}
		$$

	>  - 以上取值范围第二项都是应用等式约束情况下，考虑
		不等式约束

###	两变量二次规划求解

为叙述，记

$$\begin{align*}
g(x) & = \sum_{i=1}^N \alpha_i y_i K(x_i, x) + b \\
E_j & = g(x_j) - y_j \\
& = (\sum_{i=1}^N \alpha_i y_i K(x_i, x_j)) - y_j
\end{align*}$$

-	$g(x)$：SVM预测函数（比分类器少了符号函数）（函数间隔）
-	$E_j$：SVM对样本预测与真实值偏差

> - 以上两变量二次规划问题，沿约束方向未经剪辑解是
	$$\begin{align*}
	\alpha_2^{**} & = \alpha_2 + \frac {y_2 (E_1 - E_2)}
		\eta \\
	\eta & = K_{11} + K_{22} - 2K_{12} \\
	& = \|\phi(x_1) - \phi_(x_2)\|^2
	\end{align*}$$
	剪辑后的最优解是
	$$\begin{align*}
	\alpha_2^{*} & = \left \{ \begin{array}{l}
		H, & \alpha_2^{**} > H \\
		\alpha_2^{**}, & L \leq \alpha_2^{**} \leq H \\
		L, & \alpha_2^{*} < L
	\end{array} \right. \\
	\alpha_1^{*} & = \alpha_1 + y_1 y_2 (\alpha_2 -
		\alpha_2^{*})
	\end{align*}$$

-	记

	$$\begin{align*}
	v_i & = \sum_{j=3}^N \alpha_j y_j K(x_i, x_j) \\
	& = g(x_i) - \sum_{j=1}^2 \alpha_j y_j K(x_i, x_j) - b
		i=1,2
	\end{align*}$$

	由等式约束$\alpha_1 = (\zeta - y_2 \alpha_2) y_1$，均
	带入目标函数有

	$$\begin{align*}
	W(\alpha_1, \alpha_2) = & \frac 1 2 K_{11} (\zeta -
		y_2 \alpha_2)^2 + \frac 1 2 K_{22} \alpha_2^2 +
		y_2 K_{12} (\zeta - y_2 \alpha_2) \alpha_2 - \\
	& y_1 (\zeta - \alpha_2 y_2) - \alpha_2 +
		v_1 (\zeta - \alpha_2 y_2) + y_2 v_2 \alpha_2
	\end{align*}$$

-	对$\alpha_2$求导置0可得

	$$\begin{align*}
	(K_{11} + K_{22} - 2K_{12}) \alpha_2 & = y_2 (y_2 -
		y_1 + \zeta K_{11} - \zeta K_{12} + v_1 + v_2) \\
	& = y_2 [y_2 - y_1 + \zeta K_{11} - \zeta K_{12} +
		(g(x_1) - \sum_{j=1}^2 y_j \alpha_j K_{1j} - b) -
		(g(x_2) - \sum_{j=1}^2 y_j \alpha_j K_{2j} -b )]
	\end{align*}$$

	带入$\zeta = \alpha_1 y_1 + \alpha_2 y2$，可得

	$$\begin{align*}
	(K_11 + K_22 - 2K_{12}) \alpha_2^{**} & = y_2((K_{11} +
		K_{22} - 2K_{12}) \alpha_2 y_2 + y_2 - y_1 +
		g(x_1) - g(x_1)) \\
	& = (K_11 + K_{22} - 2K_{12}) \alpha_2 + y_2 (E_1 - E_2)
	\end{align*}$$

###	变量选择

SMO算法每个子问题的两个优化变量，其中至少一个违反KKT条件

####	外层循环--首个变量选择

在训练样本中选择违反KKT条件最严重的样本点，将对应的变量作为
第一个变量$\alpha_1$

-	检查样本点是否满足KKT条件

	$$\begin{align*}
	\alpha_i = 0 \Leftrightarrow y_i g(x_i) \geq 1 \\
	0 < \alpha_i < C \Leftrightarrow y_i g(x_i) = 1 \\
	\alpha_i = C \Leftrightarrow y_i g(x_i) \leq 1
	\end{align*}$$

-	检验过程中

	-	首先遍历所有满足条件的$0 < \alpha_i < C$样本点，即
		在间隔边界上的支持向量点

	-	若没有找到违背KKT条件的样本点，则遍历整个训练集

####	内层循环

第二个变量$\alpha_2$选择标准是其自身有足够大的变化，以加快
计算速度

-	由以上推导知，$\alpha_2^{*}$取值依赖于$|E_1 - E_2|$，
	所以可以选择$\alpha_2$使其对应的$|E_1 - E_2|$最大

	-	$\alpha_1$已经确定，$E_1$已经确定
	-	$E_1 > 0$：选最小的$E_2$
	-	$E_1 < 0$：选最大的$E_2$

-	但以上方法可能不能使得目标函数值有足够下降，采用以下
	**启发式**规则继续选择$\alpha_2$

	-	遍历间隔边界上的点
	-	遍历整个训练数据集
	-	放弃$\alpha_1$

####	更新阈值 

-	$0< \alpha_1^{*} < C$时，由KKT条件可知

	$$
	\sum_{i=1}^N \alpha_i y_i K_{i1} + b = y_1
	$$

	-	则有

		$$
		b_1^{*} = y_1 - \sum_{i=3}^N \alpha_i y_i K_{i1} -
			\alpha_1^{*} y_1 K_{11} - \alpha_2^{*} y_2 K_{21}
		$$

	-	将$E_1$定义式带入有

		$$
		b_1^{*} = -E_1 - y_1 K_{11} (\alpha_1^{*} - \alpha_1)
			- y_2 K_{21} (\alpha_2^{*} - \alpha_2) + b
		$$

-	类似的$0 < \alpha_2^{*} < C$时

	$$
	b_2^{*} = -E_2 - y_2 K_{22} (\alpha_2^{*} - \alpha_2)
		- y_1 K_{12} (\alpha_1^{*} - \alpha_1) + b
	$$

-	若

	-	$0 < \alpha_1^{*}, \alpha_2^{*} < C$，则
		$b_1{*} = b_2^{*}$

	-	$\alpha_1^{*}, $\alpha_2^{*}$均为0、C，则
		$[b_1^{*}, b_2^{*}]$中均为符合KKT条件的阈值，选择
		中点作为新阈值$b^{*}$

-	同时使用新阈值更新所有的$E_i$值

	$$
	E_i^{*} = \sum_S y_j \alpha_j K(x_i, x_j) + b^{*} - y_i
	$$

	> - $S$：所有支持向量的集合

###	算法

> - 输入：训练数据集$T$，精度$\epsilon$
> - 输出：近似解$\hat \alpha$

1.	初值$\alpha^{(0)}$，置k=0

2.	选取优化变量$\alpha_1^{(k)}, \alpha_2^{(k)}$，解析求解
	两变量最优化问题得$\alpha_1^{(k+1)}, \alpha_2^{(k+2)}$，
	更新$\alpha^{(k+1)}$

3.	若在精度范围内满足停机条件（KKT条件）

	$$\begin{align*}
	& \sum_{i=1}^N \alpha_i y_i = 0 \\
	& 0 \leq \alpha_i \leq C, i=1,2,\cdots,N \\
	& y_i g(x_i) = \left \{ \begin{array}{l}
		\geq 1, & \{x_i | \alpha_i = 0\} \\
		= 1, & \{x_i | 0 < \alpha_i < C\} \\
		\leq 1, & \{x_i | \alpha_i = C\}
	\end{array} \right. \\
	& g(x_i) = \sum_{j=1}^N \alpha_j y_j K(x_j, x_i) + b
	\end{align*}$$

	则转4，否则置k=k+1，转2

4.	取$\hat \alpha = \alpha^{(k+1)}$

