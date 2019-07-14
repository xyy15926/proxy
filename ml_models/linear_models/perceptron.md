#	*Perceptron*

-	输入：实例的特征向量
-	输出：实例类别+1、-1

##	感知机模型

感知机：线性二分类模型（判别模型）

$$
f(x) = sign(wx + b)
$$

> - $x \in \chi \subseteq R^n$：输入空间
> - $y \in \gamma \subseteq R^n$：输出空间
> - $w \in R^n, b \in R$：*weight vector*、*bias*
> - 也常有$\hat w = (w^T, b^T)^T, \hat x = (x^T + 1)^T$，
	则有$\hat w \hat x = wx + b$

-	感知机模型的假设空间是定义在特征空间的所有
	*linear classification model/linear classifier*，即函数
	集合$\{f|f(x)=wx+b\}$

-	线性方程$wx+b=0$：对应特征空间$R^n$中一个*hyperplane*

	> - $w$：超平面法向量
	> - $b$：超平面截距

	-	超平面将特征空间划分为两个部分，其中分别被分为正、负
		两类

	-	也被称为*separating hyperplane*

###	*Linearly Separable Data Set*

> - 对数据集$T=\{(x_1,y_1),\cdots,(x_N,y_N)\}$，若存在超平面
	$S: wx + b=0$能够将正、负实例点，完全正确划分到超平面
	两侧，即
	$$\begin{align*}
	wx_i + b > 0, & \forall y_i > 0 \\
	wx_i + b < 0, & \forall y_i < 0
	\end{align*}$$
	则称数据集T为线性可分数据集

##	感知机学习策略

感知机学习策略：定义适当损失函数，并将经验风险极小化，确定
参数$w, b$

###	0-1损失

经验风险：误分率（误分点总数）

-	不是参数$w, b$的连续可导函数，不易优化

###	绝对值损失

经验风险：误分类点到超平面距离

-	对误分类数据$(x_i, y_i)$，有$-y_i(wx_i + b) > 0$

-	则误分类点$(x_i, y_i)$到超平面S距离
	$$\begin{align*}
	d_i & = \frac 1 {\|w\|} |wx_i + b| \\
		& =-\frac 1 {\|w\|} y_i(wx_i + b)
	\end{align*}$$

-	则感知机损失函数可定义为
	$L(w,b) = -\sum_{x_i \in M} y_i(wx_i + b)$

	> - $M$：误分类点集合
	> - 损失函数是$w, b$的连续可导函数：使用$y_i$替代绝对值

-	损失函数$L(w,b)$梯度有

	$$\begin{align*}
	\bigtriangledown_wL(w, b) & = -\sum_{x_i \in M} y_ix_i \\
	\bigtriangledown_bL(w, b) & = -\sum_{x_i \in M} y_i
	\end{align*}$$

##	学习算法

###	*Stochastic Gredient Descent*

随机梯度下降法

> - 输入：数据集$T$、学习率$\eta, 0 \leq \eta \leq 1$
> - 输出：$w,b$、感知模型$f(x)=sgn(wx+b)$

1.	选取初值$w_0, b_0$

2.	随机选取一个误分类点$(x_i, y_i)$，即$y_i(wx_i+b) \leq 0$
	，对$w, b$进行更新

	$$\begin{align*}
	w^{(n+1)} & \leftarrow w^{(n)} + \eta y_ix_i \\
	b^{(n+1)} & \leftarrow b^{(n)} + \eta y_i
	\end{align*}$$

	> - $0 < \eta \leq 1$：*learning rate*，学习率，步长

3.	转2，直至训练集中无误分类点

> - 不同初值、随机取点顺序可能得到不同的解
> - 训练数据线性可分时，算法迭代是收敛的
> - 训练数据不线性可分时，学习算法不收敛，迭代结果发生震荡
> - 直观解释：当实例点被误分类，应该调整$w, b$值，使得分离
	超平面向**误分类点方向**移动，减少误分类点与超平面距离，
	直至被正确分类

###	学习算法对偶形式

#todo

###	算法收敛性

为方便做如下记号

> - $\hat w = (w^T, b^T)^T, \hat w \in R^{n+1}$
> - $\hat x = (x^T, 1)^T, \hat x \in R^{n+1}$

此时，感知模型可以表示为

$$
xw + b = \hat w \hat x = 0
$$

> -	数据集$T={(x_1, y_1), (x_2, y_2),...}$线性可分，其中：
	$x_i \in \mathcal{X = R^n}$，
	$y_i \in \mathcal{Y = \{-1, +1\}}$，则

> > -	存在满足条件$\|\hat w_{opt}\|=1$超平面
		$\hat w_{opt} \hat x = 0$将训练数据完全正确分开，且
		$\exists \gamma > 0, y_i(\hat w_{opt} x_i) \geq \gamma$

> > -	令$R = \arg\max_{1\leq i \leq N} \|\hat x_i\|$，则
		随机梯度感知机误分类次数$k \leq (\frac R \gamma)^2$

####	超平面存在性

-	训练集线性可分，存在超平面将训练数据集完全正确分开，可以
	取超平面为$\hat w_{opt} \hat x = 0$

-	令$\|\hat w_{opt}\| = 1$，有

	$$\forall i, y_i(\hat w_{opt} \hat x_i) > 0$$
	
	可取

	$$\gamma = \min_i \{ y_i (\hat w_{opt} \hat x) \}$$

	满足条件

####	感知机算法收敛性

-	给定学习率$\eta$，随机梯度下降法第k步更新为
	$\hat w_k = \hat w_{k-1} + \eta y_i \hat x_i$

-	可以证明

	-	$\hat w_k \hat w_{opt} \geq k\eta\gamma$

		$$\begin{align*}
		\hat w_k \hat w_{opt} & =
			\hat w_{k-1} \hat w_{opt} +
				\eta y_i \hat w_{opt} \hat x_i \\ 
			& \geq \hat w_{k-1} \hat w_{opt} +
				\eta\gamma \\
			& \geq k\eta\gamma
		\end{align*}$$

	-	$\|\hat w_k\|^2 \leq k \eta^2 R^2$

		$$\begin{align*}
		\|\hat w_k\|^2 & = \|\hat w_{k-1} +
			\eta y_i x_i \|^2 \\
		& = \|\hat w_{k-1}\|^2 + 2\eta y_i \hat w_{k-1}
			\hat x_i + \eta^2 \|\hat x_i\|^2 \\
		& \leq \|w_{k-1}\|^2 + \eta^2 \|\hat x_i\|^2 \\
		& \leq \|w_{k-1}\|^2 + \eta^2 R^2 \\
		& \leq k\eta^2 R^2
		\end{align*}$$

-	则有

	$$\begin{align*}
	k \eta \gamma & \leq \hat w_k \hat w_{opt} \leq
		\|\hat w\| \|\hat w_{opt}\| = \|\hat w\|
		\leq \sqrt k \eta R \\
	k^2 \gamma^2 & \leq k R^2
	\end{align*}$$

> - 直观理解就是超平面**最大移动次数**不大于**最大移动距离**
	除以**最小移动步长**
> > -	$\eta \gamma^2$：超平面法向量最小增加量（移动步长）
> > -	$\eta R^2$：超平面法向最大增加量（移动距离）
> > -	但是超平面不可能将所有点都归为同一侧

> - 误分类次数有上界，经过有限次搜索可以找到将训练数据完全
	正确分开的分离超平面，即训练数据集线性可分时，算法的迭代
	形式是收敛的


