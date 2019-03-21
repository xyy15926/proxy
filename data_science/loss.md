#	Loss Function

##	Singluar Loss

单个样本点损失：度量模型“一次”预测的好坏

-	最基本损失，之和的损失都建立在此基础之上
-	模型在单点上的性质

###	0-1 Loss

0-1损失函数

$$
L(Y, f(x)) = \left \{ \begin{align*}
	1, & Y \neq f(X) \\
	0, & Y = f(X)
\end{align*} \right.
$$

-	适用场合
	-	二分类：Adaboost
	-	多分类：Adaboost.M1

### Hinge Loss

合页损失函数

$$\begin{align*}
L(Y, f(X)) & = [1 - Yf(x)]_{+} \\
[Z]_{+} & = \left \{ \begin{align*}
	Z, & Z > 0 \\
	0, & Z \leq 0
\end{align*} \right.
\end{align*}$$

> - $Y \in \{-1, +1\}$

-	合页损失函数是0-1损失函数的上界
-	合页损失函数要求分类不仅正确，还要求确信度足够高损失才
	为0，即对学习有更高的要求
-	适用场合
	-	二分类：线性支持向量机

###	Quadratic Loss

平方损失函数

$$
L(Y, f(X)) = (Y - f(X))^2
$$

-	适用场合
	-	回归预测：线性回归

###	Absolute Loss

绝对损失函数

$$
L(Y, f(X)) = |Y-f(X)|
$$

-	适用场合
	-	回归预测

###	Logarithmic Loss

对数损失函数（对数似然损失函数）

$$
L(Y, P(Y|X)) = -logP(Y|X)
$$

-	适用场合
	-	多分类：贝叶斯生成模型

###	Exponential Loss

指数函数函数

$$
L(Y, f(x)) = exp\{-yf(x)\}
$$

-	适用场合
	-	二分类：前向分步算法

###	Pseudo Loss

伪损失：考虑个体$(x_i, y_i)$

-	$h(x_i, y_i)=1, \sum h(x_i, y)=0$：完全正确预测
-	$h(x_i, y_i)=0, \sum h(x_i, y)=1$：完全错误预测
-	$h(x_i, y_i)=1/M$：随机预测（M为分类数目）

据此构造伪损失

$$\begin{align*}
L(Y, f(x)) & = \frac 1 2 \sum_{(i,y) \in B} D_{i,y}
	(1 - h(x_i, y_i) + h(x_i, y)) \\
& = \frac 1 2 \sum_{i=1}^N D_i (1 - h(x_i, y_i) +
	\sum_{y \neq y_i} (w_{i,y} h(x_i, y)))
\end{align*}$$

> - $D$：权重分布（行和为1，但不满足列和为1）
> > -	$D_{i,y}$：个体$x_i$中错误标签$y$的权重，代表从个体
		$x_i$中识别出错误标签$y$的重要性
> - $w$：个体各错误标签权重边际分布
> - $h(x, y)$：模型$h$预测样本$x$为$y$的置信度
> > -	$h(x_i,y_i)$：预测正确的置信度
> > -	$h(x_i,y), y \neq y_i$：预测$x_i$为错误分类$y$置信度

-	伪损失函数考虑了样本和**标签**的权重分布
	-	通过改变此分布，能够更明确的关注难以预测的个体标签，
		而不仅仅个体

-	伪损失随着分类器预测准确率增加而减小
	-	分类器$h$输出向量各分量相同时，伪损失最大达到0.5，
		此时 就是随机预测
	-	伪损失大于0.5时，应该将使用$1-h$

-	适用场景
	-	多分类：Adaboost.M2

##	Total Loss

模型（目标函数）在样本整体的损失：度量模型整体预测效果

-	代表模型在整体上的性质

-	可以用于**设计学习策略、评价模型**
	-	风险函数
	-	评价函数

-	有时在算法中也会使用整体损失

###	*Risk Function*

（期望）风险函数：*expected loss*，是损失函数$L(Y, f(X))$
（随机变量）期望

$$
R_{exp}(f) = E_p[L(Y, f(X))] = \int_{x*y} L(y,f(x))P(x,y) dxdy
$$

> - $P(X, Y)$：随机变量$(X, Y)$遵循的联合分布，未知


-	风险函数值度量模型预测错误程度

-	评价标准（**监督学习目标**）就应该是选择期望风险最小

-	联合分布未知，所以才需要学习，否则可以直接计算条件分布
	概率，而计算期望损失需要知道联合分布，因此监督学习是一个
	病态问题

###	*Empirical Risk*

经验风险：*empirical loss*，模型关于给定训练数据集的平均损失

$$
R_{emp}(f) = \frac 1 N \sum_{i=1}^N L(y_i, f(x_i;\theta))
$$

> - $\theta$：模型参数
> - 经验风险损失一般是$f(x)$模型的函数，建立**损失函数和
	模型参数之间的函数关系**

-	根据大数定律，样本量容量N趋于无穷时，$R_{emp}(f)$趋于
	$R_{exp}(f)$

-	但是现实中训练样本数目有限、很小，利用经验风险估计期望
	常常并不理想，需要对经验风险进行矫正

-	例子
	-	*maximum probability estimation*：极大似然估计
		-	模型：条件概率分布
		-	损失函数：对数损失函数

###	*Structual Risk*

*结构风险*：在经验风险上加上表示**模型复杂度**的
*regularizer*（*penalty term*）

$$
R_{srm} = \frac 1 N \sum_{i=1}^N L(y_i, f(x_i)) +
	\lambda J(f)
$$

> - $J(f)$：模型复杂度，定义在假设空间$F$上的泛函
> - $\lambda$：权衡经验风险、模型复杂度的系数

-	结构风险最小化通过添加*regularization*（正则化）实现

-	模型复杂度$J(f)$表示对复杂模型的惩罚：模型$f$越复杂，
	复杂项$J(f)$越大

-	例子
	-	*maximum posterior probability estimation*：最大后验
		概率估计
		-	损失函数：对数损失函数
		-	模型复杂度：模型先验概率对数后取负
		-	先验概率对应模型复杂度，先验概率越小，复杂度越大

##	Batch Loss

模型（目标函数）在某个batch上的损失

-	是模型在batch上的特征，对整体的代表性取决于batch大小

	-	batch越大对整体代表性越好
	-	batch大小为1时，就是某个样本点个体损失
	-	batch大小为整个训练集时，就是经验（结构）风险

-	这个loss是学习算法中最常用的loss

	-	虽然策略往往是风险最小化，但在实际操作中往往是使用
		batch loss替代风险（参见*algorithms*）
	-	所以和风险一样可能会带有正则化项

	-	损失极值：SVM（几何间隔最小）

##	*regularization*

正则化：（向目标函数）添加额外信息以求解病态问题、避免过拟合

-	常应用在机器学习、逆问题求解

	-	对模型（目标函数）复杂度惩罚
	-	提高学习模型的泛化能力、避免过拟合
	-	学习简单模型：稀疏模型、引入组结构

-	有多种用途

	-	最小二乘也可以看作是简单的正则化
	-	岭回归中的$\mathcal{l_2}$范数

###	模型复杂度

模型复杂度：经常作为正则化项添加作为额外信息添加的，衡量模型
复杂度方式有很多种

-	函数光滑限制

	-	多项式最高次数

-	向量空间范数

	-	$\mathcal{L_0}$ norm：参数个数
	-	$\mathcal{L_1}$ norm：参数绝对值和
	-	$\mathcal{L_2}$ norm：参数平方和

####	$\mathcal{L_0}$ norm

-	稀疏化约束

-	解$\mathcal{L_0}$范数正则化是NP-hard问题

####	$\mathcal{L_1}$ norm

-	$\mathcal{L_1}$范数可以通过凸松弛得到$\mathcal{L_0}$的
	近似解

-	有时候出现解不唯一的情况

-	$\mathcal{L_1}$范数凸但不严格可导，可以使用依赖次梯度的
	方法求解极小化问题

-	应用
	-	*LASSO*

-	求解
	-	*Proximal Method*
	-	*LARS*

####	$\mathcal{L_2}$ norm

-	$\mathcal{L_2}$范数凸且严格可导，极小化问题有解析解

-	求解

####	$\mathcal{L_1 + L_2}$

-	有组效应，相关变量权重倾向于相同

-	应用
	-	*Elastic Net*

###	Earlty Stopping

*Early Stopping*也可以被视为是*regularizing on time*

-	迭代式训练随着迭代次数增加，往往会有学习复杂模型的倾向
-	对时间施加正则化，可以减小模型复杂度、提高泛化能力

###	稀疏解产生

稀疏解：待估参数系数在某些分量上为0

####	$\mathcal{L_1}$稀疏解的产生

$\mathcal{L_1}$范数在参数满足**一定条件**情况下，能对
**平方损失**产生稀疏效果

-	在$[-1,1]$内$y=|x|$导数大于$y=x^2$（除0点），所以特征在
	一定范围内变动时，为了取到极小值，参数必须始终为0

-	满足条件的**一定范围**就是，特征满足在0附近、$y=x$导数
	较大

	-	高阶项在0点附近增加速度较慢，所以$\mathcal{L_1}$
		能产生稀疏解是很广泛的

	-	$mathcal{L_1}$前系数越大，能够容许高阶项增加的幅度
		越大，即压缩能力越强
	
-	在0附近导数“不小”，即导数在0点非0

	-	对多项式正则化项而言，其必须“带有”$\mathcal{L_1}$，
		并且稀疏化解就是$\mathcal{L_1}$起决定性作用，其他项
		没有稀疏解的用途

	-	对“非多项式”正则化项，比如：$e^{|x|}-1$、$ln(|x|+1)$
		，在0点泰勒展开同样得到$\mathcal{L_1}$，但是这样的
		正则化项难以计算数值，所以其实不常用

####	$\mathcal{L_1}$稀疏解推广

推广的方向有如下

-	正负差异化：在正负设置不同大小的$\mathcal{L_1}$，赋予
	在正负不同的压缩能力，甚至某侧完全不压缩

-	分段函数压缩：即只要保证在0点附近包含$\mathcal{L_1}$用于
	产生稀疏解，远离0处可以设计为常数等不影响精确解的值

	-	*smoothly clipped absolute deviation*

		$$
		R(x|\lambda, \gamma) = \left \{ \begin{array} {l}
			\lambda|x| \qquad & if |x| \leq \lambda \\
			\frac {2\gamma\lambda|x| - x^2 - {\lambda}^2 }
				{2(\gamma - 1)} &
				if \gamma< |x| <\gamma\lambda \\
			\frac {{\lambda}^2(\gamma+1)} 2 &
				if |x| \geq \gamma\lambda
		\end{array} \right.
		$$

	-	derivate of SCAD

		$$
		R(x; \lambda, \gamma) = \left \{ \begin{array} {l}
			\lambda \qquad & if |x| \leq \gamma \\
			\frac {\gamma\lambda - |x|} {\gamma - 1} &
				if \lambda < |x| < \gamma\lambda \\
			0 & if |x| \geq \gamma\lambda
		\end{array} \right.
		$$

	-	*minimax concave penalty*

		$$
		R_{\gamma}(x;\lambda) = \left \{ \begin{array} {l}
			\lambda|x| - \frac {x^2} {2\gamma} \qquad &
				if |x| \leq \gamma\lambda \\
			\frac 1 2 \gamma{\lambda}^2 &
				if |x| > \gamma\lambda
		\end{array} \right.
		$$

-	分指标：对不同指标动态设置$\mathcal{L_0}$系数

	-	*adaptive lasso*：$\lambda \sum_J w_jx_j$

####	稀疏本质

稀疏本质：极值、**不光滑**，即导数符号突然变化

-	若某约束项导数符号突然变化、其余项在该点处导数为0，为
	保证仍然取得极小值，解会聚集（极小）、疏远（极大）该点
	（类似坡的陡峭程度）

	-	即这样的不光滑点会**抑制解的变化**，不光滑程度即导数
		变化幅度越大，抑制解变化能力越强，即吸引、排斥解能力
		越强
	-	这样非常容易构造压缩至任意点的约束项
	-	特殊的，不光滑点为0时，即得到稀疏解

-	可以设置的多个极小不光滑点，使得解都在不连续集合中

	-	可以使用三角函数、锯齿函数等构造，不过需要考虑的是
		这样的约束项要起效果，必然会使得目标函数非凸

		-	但是多变量场合，每个变量实际解只会在某个候选解
			附近，其邻域内仍然是凸的
		-	且锯齿函数这样的突变非凸可能和凸函数具有相当的
			优秀性质

	-	当这些点均为整数时，这似乎可以近似解**整数规划**




