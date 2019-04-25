#	Proximal Gredient Method

##	原始近端

近端算法：用于解*non-differentiable*凸优化问题的通用投影形式

###	问题

$$
\min_{x \in R^N} \sum_{i=1}^N f_i(x)
$$

> - $f_i(x)$：凸函数，不一定处处可微

-	目标函数中包含不处处连续可微函数，整个目标函数不光滑

	-	所以无法使用传统的光滑优化手段，如：最速下降、共轭
		梯度

	-	光滑凸函数极小化衡量标准$\bigtriangledown(F+R)(x)=0$
		变为

		$$0 \in \partial (F+R)(x)$$

		> - $\partial \phi$：实凸函数的*subdifferential*

-	近端算法：分开考虑各个函数，对非光滑函数使用近端算子处理

-	以下算法都是近端算法的实例

	-	*shrinkage thresholding algorithm*
	-	*projected Landweber*
	-	*projected gradient*
	-	*alternating projections*
	-	*alternating-directions method of multipliers*
	-	*alternating split Bregman*

###	*Porjection Operator*

投影算子

$$\begin{align*}
proj_C(x) & = \arg\min_{y \in C} \|y - x\|^2 \\
& = \arg\min_{y \in R^N} l_C(x) + \frac 1 2 \|y-x\|^2
\end{align*}$$

-	点$x$在凸集$C$上的投影：$X$上距离$x$的欧式距离最近的点

###	*Proximal Operator*

$$
prox_{f}(x) = \mathop \arg\min_y (f(y)+\frac 1 2 \|x-y\|^2)
$$

> - $f(x)$：凸函数

###	*Projection & Proximal*

考虑非空凸集$C \subseteq R^N$，示性函数的近端算子为

$$\begin{align*}
prox_{l_C}(x) & = \mathop \arg\min_y \left \{ \begin{array}{l}
	\frac 1 2 \|x-y\|^2, & y \in C \\
	+ \infty, & y \notin C \\
\end{array} \right. \\
& = \arg\min_{y \in C} \frac 1 2 \|x-y\|^2 \\
& = P_C(x)
\end{align*}$$

> - C示性函数、投影、距离定义参见*convex*

-	**近端算子是投影算子的推广**
-	一般的近端算子是**到函数的距离**???

###	*Subgredient*

-	对一般的凸函数$f$，其近端算子满足

	$$
	p = prox_f(x) \Leftrightarrow x - p \in \partial f(p)
		\quad (\forall (x,p) \in R^N * R^N)
	$$

-	对光滑凸函数$f$，上述等式对其近端算子约简为

	$$
	p = prox_f(x) \Leftrightarrow x-p = \bigtriangledown f(p)
	$$

##	近端算法

目标函数

$$
\min_{x \in \mathcal{H}}F(x) + R(x)
$$

> - $F(x)$：*Lipschitz continous*、可微的凸函数
> - $R(x)$：下半连续凸函函数，可能不光滑
> - $\mathcal{H}$：集合，如：希尔伯特空间

###	近端算子

-	对$F(x)+R(x)$在$x_0$附近作泰勒展开
	$$
	F(x)+R(x) \leq F(x_0) + (x-x_0)^2\bigtriangledown F(x_0)
		+ \frac 1 {2\lambda} \|x-x_0\|^2 + R(x)
	$$

	> - $\lambda \in (0, \frac 1 L]$
	> - $L$：Lipschitz常数
	> - $\leq, L$：由Lipschitz连续可取

	-	所以不等式右边就是$F(x)+R(x)$的一个上界
	-	可以对将对其求极小化转化对此上界求极小

-	对不等式右边添加常数项
	$\frac \lambda 2 \|\bigtriangledown F(x_0)\|^2$凑完全
	平方项得近端算子

	$$
	prox_{\lambda R}(x_0) = \mathop \arg\min_x (R(x) +
		\frac 1 {2\lambda} \|x-v\|_2^2)
	$$

	> - $v = x_0 - \lambda \bigtriangledown F(x_0)$
	> - 泰勒展开是局部性质，所以这里$v$中肯定不能一步到位
		直接取$\lambda$，而是使用另一个较小的参数$\gamma$，
		控制搜索范围
	> - 这里近端算子里的$\mathcal{L_2}$范数系数和标准的不同

-	#todo，这里好像还是有问题，1. 推出的近端算子是原始近端
	算子的推广？2. $\lambda, \gamma$系数约束问题

![proximal_operator](imgs/proximal_operator.png)

-	寻找距离点$v$不太远、$f(x)$尽可能小的$x$

-	对目标函数不是处处连续可微的情况，通常使用*subgredient*
	进行的优化，而次梯度会导致

	-	求解速度慢
	-	通常不会产生稀疏解

###	算法

-	*Gradient Step*：沿着$F(x)$负梯度方向寻找下个点

	$$
	v^{(t+1)} = x^{(t)} - \gamma \bigtriangledown F(x^{(t)})
	$$

-	*Proximal Operator Step*：使用近端算子优化

	-	$F(x)$的Lipschitz continuous系数已知为L，可取
		$lambda \in (0, \frac 1 L)$

		$$\begin{align*}
		x^{(t+1)} & = prox_{\lambda R}(v^t) \\
		& = \mathop \arg\min_x (R(x) + \frac 1 {2\lambda}
			\|x-v\|_2^2)
		\end{align*}$$

	-	否则的使用line search方法寻找

		> - $$z = prox_{\lambda R}(v^t)$$
		> - $$
			F^{'}(z) = F(v^t) + \bigtriangledown F^T(v^t)(z-v)
				+ \frac 1 {2\lambda} \|v^t-z\|^2
			$$
		> - 若$F(z) \leq F^{'}(z)$，停止取$x^{(t+1)}=z$，
			否则取$\lambda = \frac 1 2 \lambda$继续迭代

> - 这里Lipschitz连续系数L是啥？如果是导数上界，为啥在
	line search方法中会作为平方项系数
> - [ref](http://www.luolei.info/2016/09/27/proximalAlgo/)

##	*Alternating Projection Method*

*POCS/project onto convex sets method*：用于解同时满足多个
凸约束的算法

-	$f_i$作为非空闭凸集$C_i$示性函数，表示一个约束，则整个
	问题约简为*convex feasibility problem*

-	只需要找到位于所有$C_i$交集的解即可

-	每次迭代

	$$
	x^{(k+1)} = P_{C_1}P_{C_2} \cdots P_{C_n}x_k
	$$

> - 在其他问题中投影算子不再适合，需要更一般的算子，在其他
	各种同样的凸投影算子中，近端算子最合适






