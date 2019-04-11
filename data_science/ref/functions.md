#	常用函数

##	通用

-	`I`：示性/指示函数
	-	满足条件时取1，否则取0

-	`sign`：符号函数
	-	`>0`：取1
	-	`<0`：取-1

##	核函数

###	*Uniform Kernel*

$$
K(d) = \frac 1 2 * I(|d|<1)
$$

###	*Triangle Kernel*

$$
K(d) = (1-|d|) * I(|d|<1)
$$

###	*Epanechnikov Kernel*

$$
K(d) = \frac 3 4 (1-d^2) * I(|d|<1)
$$

###	*Quartic Kernel*

$$
K(d) = \frac {15} {16} (1-d^2)^2 * I(|d|<1)
$$

###	*Triweight Kernel*

$$
K(d) = \frac {35} {32} (1-d^2)^3 * I(|d|<1)
$$

###	*Gauss Kernel*

$$
K(d) = \frac 1 {\sqrt{2\pi}} exp(- \frac {d^2} 2) * I(|d|<1)
$$

###	*Cosine Kernel*

$$
K(d) = \frac \pi 4 cos(\frac \pi 2) * I(|d|<1)
$$

##	距离函数

$dist(x,y)$：不一定是空间距离，应该认为是两个对象x、y之间的
**相似程度**

-	距离和相似度是互补的
-	可以根据处理问题的情况，自定义距离

###	单点距离

####	*Minkowski Distance*

闵科夫斯基距离：向量空间$\mathcal{L_p}$范数

$$
d_{12} = \sqrt [1/p] {\sum_{k=1}^n |x_{1,k} - x_{2,k}|^p}
$$

-	表示一组距离族

	-	$p=1$：*Manhattan Distance*，曼哈顿距离
	-	$p=2$：*Euclidean Distance*，欧式距离
	-	$p \rightarrow \infty$：*Chebychev Distance*，
		切比雪夫距离

-	闵氏距离缺陷

	-	将各个分量量纲视作相同
	-	未考虑各个分量的分布

####	*Mahalanobis Distance*

马氏距离：表示数据的协方差距离

$$
d_{12} = \sqrt {({x_1-\mu}^T) \Sigma^{-1} (x_2-\mu)}
$$

> - $\Sigma$：总体协方差矩阵

-	优点

	-	马氏距离和原始数据量纲无关
	-	考虑变量相关性

-	缺点

	-	需要知道总体协方差矩阵，使用样本估计效果不好

####	*LW Distance*

兰氏距离：*Lance and Williams Distance*，堪培拉距离

$$
d_{12} = \sum^{n}_{k=1} \frac {|x_{1,k} - x_{2,k}|}
	{|x_{1,k} + x_{2,k}|}
$$

-	对接近0的值非常敏感
-	对量纲不敏感
-	未考虑变量直接相关性，认为变量之间相互独立

####	*Consine Similarity*

余弦相似度

$$
simimarity = cos(\theta) = \frac {x_1 x_2} {\|x_1\|\|x_2\|}
$$

###	定性变量距离

####	差异程度

$$
diff = \frac 1 p \sum_{i=1}^p  (v^{(1)}_i - v^{(2)}_i)^k 
$$

> - $v_i$：虚拟变量
> - $p$：虚拟变量数量

###	组间距离

####	*Single Linkage*

####	*Average Linkage*

####	*Complete Linkage*

###	符号距离

> - $T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_n,y_n)\}$：样本点集
> - $wx + b = 0$：超平面

####	*Functional Margin*

函数间隔

$$
\hat{\gamma_i} = y_i(wx_i + b)
$$


-	函数间隔可以表示分类的正确性、确信度

	-	正值表示正确
	-	间隔越大确信度越高

-	点集与超平面的函数间隔取点间隔最小值
	$\hat{T} = \min_{i=1,2,\cdots,n} \hat{\gamma_i}$

-	超平面参数$w, b$成比例改变时，平面未变化，但是函数间隔
	成比例变化

####	*Geometric Margin*

几何间隔

$$\begin{align*}
\gamma_i & = \frac {y_i} {\|w\|} (wx_i + b) \\
	& = \frac {\hat \gamma_i} {\|w\|}
\end{align*}$$

-	几何间隔一般是样本点到超平面的*signed distance*

	-	点正确分类时，几何间隔就是点到直线的距离

-	几何间隔相当于使用$\|w\|$对函数间隔作规范化

	-	$\|w\|=1$时，两者相等
	-	几何间隔对确定超平面、样本点是确定的，不会因为超平面
		表示形式改变而改变

-	点集与超平面的几何间隔取点间隔最小值
	$\hat{T} = \min_{i=1,2,\cdots,n} \hat{\gamma_i}$

##	激活函数

###	指数类

####	Sigmoid

将实数映射到(0, 1)区间

$$
sigmoid(x) = \frac 1 {1+e^{-x}}
$$

-	用途
	-	隐层神经元输出
	-	二分类输出

-	缺点
	-	激活函数计算量大，BP算法求误差梯度时，求导涉及除法
	-	误差反向传播时容易出现梯度消失
	-	函数收敛缓慢

####	Hard_Sigmoid

计算速度比sigmoid激活函数快

$$
hard_signmoid(x) = \left \{ \begin {array} {l}
	0 & x < -2.5 \\
	1 & x > 2.5 \\
	0.2*x + 0.5 & -2.5 \leq x \leq 2.5 \\
\end {array} \right.
$$

####	Softmax

主要用于多分类神经网络输出

$$
softmax(x_i) = \frac {e^{x_i}} {\sum_{k=1}^K e^{x_k}}
$$

-	使用指数
	-	拉开数值之间差距，大者更大，小者更小
	-	保证激活函数可导

-	Softmax回归参数冗余

####	Softplus

$$
softplus(x) = log(exp(x)+1)
$$

####	Tanh

双曲正切函数

$$
\begin{align*}
tanh(x) & = \frac {sinhx} {coshx} \\
	& = \frac {e^x - e^{-x}} {e^x + e^{-x}} \\
\end{align*}
$$

> - $\frac{\partial tanh(x)}{\partial x} = (1 - tanh(x))^2$
	：非常类似普通正切函数，可以简化梯度计算

###	线性指数类

####	Elu

线性指数

$$
elu(x, \alpha) =
\left \{ \begin{array} {l}
	x & x > 0 \\
	\alpha (e^x - 1) & x \leqslant 0 \\
\end{array} \right.
$$

####	Selu

可伸缩指数线性激活：可以两个连续层之间保留输入均值、方差

-	正确初始化权重：`lecun_normal`初始化
-	输入数量足够大：`AlphaDropout`
-	选择合适的$\alpha, scale$值

$$
selu(x) = scale * elu(x, \alpha)
$$

###	线性类

####	Softsign

$$
softsign(x) = \frac x {abs(x) + 1)}
$$

####	ReLU

*Rectfied Linear Units*：修正线性单元

$$
relu(x, max) = \left \{ \begin{array} {l}
	0 & x \leq 0 \\
	x & 0 < x < max \\
	max & x \geq max \\
\end {array} \right.
$$

####	LeakyReLU

*Leaky ReLU*：带泄露的修正线性

$$
relu(x, \alpha, max) = \left \{ \begin {array} {l}
	\alpha x & x \leq 0 \\
	x & 0 < x < max \\
	max & x \geq max \\
\end {array} \right.
$$

####	PReLU

*Parametric ReLU*：参数化的修正线性

$$
prelu(x) = \left \{ \begin{array} {l}
	\alpha x & x < 0 \\
	x & x> 0 \\
\end{array} \right.
$$

> - *$\alpha$*：自学习参数（向量），需要给出权重初始化方法
	（正则化方法、约束）

####	ThreshholdReLU

带阈值的修正线性

$$
threshhold_relu(x, theta)= \left \{ \begin{array} {l}
	x & x > theta \\
	0 & otherwise \\
\end{array} \right.
$$

####	Linear

线性激活函数：不做任何改变

###	梯度消失

激活函数导数太小（$<1），压缩**误差**变化


