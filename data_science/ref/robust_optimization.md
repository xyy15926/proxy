#	*Robust Optimization*

##	背景

稳健优化：利用凸理论、对偶理论中概念，使得凸优化问题中的解
对参数的*bounded uncertainty*有限不确定性（波动）不敏感

-	稳健优化在机器学习涉及方面：不确定优化、过拟合

	-	*Connecting Consistency*
	-	*Generalization Ability*
	-	*Sparsity*
	-	*Stability*

-	不确定性来源

	-	模型选择错误
	-	假设不成立
	-	忽略必要因素
	-	经验分布、函数无法正确估计整体分布

-	过拟合判断依据

	-	*metric entropy*
	-	*VC-dimension*

###	对比

优化问题对问题参数的扰动非常敏感，以至于解经常不可行、次优

-	*Stochastic Programming*：使用概率描述参数不确定性，

-	稳健优化则假设问题参数在某个给定的先验范围内随意变动

	-	不考虑参数的分布

	-	利用概率论的理论，而不用付出计算上的代价

###	策略（最优化问题）
$$
\begin{align*}
\min_x & : f_0(x) \\
s.t. & : f_i(x) \leq 0, i=1,2,\cdots,m
\end{align*}
$$

$$
\begin{align*}
\min_x & : f_0(x) \\
s.t. & : f_i(x, u_i) \leq 0, \forall u_i \in \mathcal{U}_i,
	i=1,2,\cdots,m
\end{align*}
$$

-	$\mathcal{U}_i $：*uncertainty set*，不确定集

###	*Computational Tractablity*

稳健优化易解性：在满足标准或一点违反
*Slater-like regularity conditions*情况下，求解稳健优化问题
等同于求解对以下凸集$\mathcal{X(U)}$的划分（求出凸集）

$$
\mathcal{X(U)} \overset {\triangle} {=}
	\{ x: f_i(x, u_i) \leq 0,
	\forall u_i \in \mathcal{U}_i, i=1,2,\cdots,m \}
$$

-	若存在高效算法能确定$x \in \mathcal{X(U)}$、或者能够提供
	分离超平面，那么问题可以在多项式时间中求解

-	即使所有的约束函数$f_i$都是凸函数，此时$\mathcal{X(U)}$
	也是凸集，也有可能没有高效算法能够划分出$\mathcal{X(U)}$

-	然而在大部分情况下，稳健化后的问题都能高效求解下，和原
	问题复杂度相当

####	复杂度说明

-	LP + Polyhedra Uncertainty：LP
-	LP + Ellipsoidal Uncertainty：SOCP
-	CQP + Ellipsoidal Uncertainty：SDP
-	SDP + Ellipsoodal Uncertainty：NP-hard

> - *LP*：Linear Program，线性规划
> - *SOCP*：Second-Order Cone Program，二阶锥规划
> - *CQP*：Convex Quadratic Program，凸二次规划
> - *SDP*：Semidefinite Program，半定规划
> - *Polyhedra Uncertainty*：多项式类型不确定
> - *Ellipsodial Uncertainty*：椭圆类型不确定
> - *NP-hard*：NP难问题，至少和NPC问题一样困难得问题

####	Example

*Linear Programs with Polyhedral Uncertainty*

###	概率解释、结果

-	稳健优化的计算优势很大程度上来源于，其形式是固定的，不再
	需要考虑概率分布，只需要考虑不确定集

-	计算优势使得，即使不确定性是随机、且分布已知，稳健优化
	仍然具有吸引力

-	在一些概率假定下，稳健优化可以给出稳健化问题解的某些
	概率保证，如：可行性保证（在给定约束下，解能以多大概率
	不超过约束）

###	*Uncertainty Set*

####	*Atomic Uncertainty Set*

原子不确定集

$$
\begin{align*}
(I) & 0 \in \mathcal{U}_0 \\
(II) & \forall w_0 \in R^n: \sup_{u \in \mathcal{U}_0
	[-w_0^T u^{'} < +\infty
\end{align}
$$

##	Robust Optimization and Adversary Resistant Learning

即稳健优化在机器学习中处理不确定性（随机的、对抗性的）

-	稳健优化中在机器学习中应用

-	稳健学习在很多学习任务中都有提出

	-	学习和规划
	-	Fisher线性判别分析
	-	PCA

这里考虑经典的**二分类软阈值SVM**

$$
\begin{align*}
\min_{w,b,\xi}: \quad & \mathcal{ r(w,b) +
	C\sum_{i=1}^m \xi_i} \\
s.t.: & \xi_i \geq [1-y_i(<w,x_i> + b)], i=1,\cdots,m; \\
	& \xi_i \geq 0, i=1,\cdots,m;
\end{align*}
$$

###	Corrupted Location

-	椭圆不确定集：随机导致的

-	正则化项使用

	-	传统的二范数，一范数同样使用的稀疏的解
	
-	概率解释：风险控制

###	Missing Data

-	多项式不确定：对抗删除数据（alpha go）

-	使用无效特征消去偏置

-	对max损失取对偶得到min带入得到SOCP

##	Robust Optimization and Regularization

-	统一从稳健优化的角度解释学习算法中的优秀性质

	-	正则化
	-	稀疏
	-	一致性

-	指导寻找新的算法

	-	大数定理、中心极限定理表明即使各个特征上随机不确定项
		是独立的，其本身也会有强烈的耦合倾向，表现出相同特征
		、像会相互影响一样

	-	这促使寻找新的稳健算法，其中随机不确定项是耦合的

###	SVM

-	



