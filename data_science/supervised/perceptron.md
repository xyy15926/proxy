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

-	感知机模型的假设空间是定义在特征空间的所有
	*linear classification model/linear classifier*，即函数
	集合$\{f|f(x)=wx+b\}$

-	线性方程$wx+b=0$：对应特征空间$R^n$中一个*hyperplane*

	> - $w$：超平面法向量
	> - $b$：超平面截距

	-	超平面将特征空间划分为两个部分，其中分别被分为正、负
		两类

	-	也被称为*separating hyperplane*

> - *linearly separable data set*：存在超平面$S: wx + b=0$
	能够将正、负实例点，完全正确划分到超平面两侧的数据集，
	即$\all y_i=+1, wx_i + b > 0$

###	感知机学习策略

感知机学习策略：定义适当损失函数，并将经验风险极小化，确定
参数$w, b$

####	0-1损失

经验风险：误分率（误分点总数）

-	不是参数$w, b$的连续可导函数，不易优化

####	绝对值损失

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

###	学习算法

####	*Stochastic Gredient Descent*

随机梯度下降法：不同初值、随机取点顺序可能得到不同的解

-	损失函数梯度

	$$\begin{align*}
	\bigtriangledown_wL(w, b) & = -\sum_{x_i \in M} y_ix_i \\
	\bigtriangledown_bL(w, b) & = -\sum_{x_i \in M} y_i
	\end{align*}$$

-	任意选择$w_0, b_0$，确定超平面

-	随机选取一个误分类点$(x_i, y_i)$，对$w, b$进行更新

	$$\begin{align*}
	w^{(n+1)} & \leftarrow w^{(n)} + \eta y_ix_i \\
	b^{(n+1)} & \leftarrow b^{(n)} + \eta y_i
	\end{align*}$$

	> - $0 < \eta \leq 1$：*learning rate*，学习率，步长

-	不断迭代，使得损失函数不断减小，直至为0





