#	激活函数

##	指数类

##	Sigmoid

将实数映射到(0, 1)区间

$$
sigmoid(z) = \frac 1 {1+e^{-z}}
$$

> - $z= wx+b$

-	用途
	-	隐层神经元输出
	-	二分类输出

-	缺点
	-	激活函数计算量大，BP算法求误差梯度时，求导涉及除法
	-	误差反向传播时容易出现梯度消失
	-	函数收敛缓慢

###	Hard_Sigmoid

计算速度比sigmoid激活函数快

$$
hard_signmoid(z) = \left \{ \begin {array} {l}
	0 & z < -2.5 \\
	1 & z > 2.5 \\
	0.2*z + 0.5 & -2.5 \leq z \leq 2.5 \\
\end {array} \right.
$$

> - $z= wx+b$

###	Softmax

主要用于多分类神经网络输出

$$
softmax(z_i) = \frac {e^{z_i}} {\sum_{k=1}^K e^{z_k}}
$$

> - $z_i = w_i x + b_i$：$(w_i, b_i)$组数同分类数量，和输入
	$x$维度无关

> - $K$：分类数目

-	使用指数
	-	拉开数值之间差距，大者更大，小者更小
	-	保证激活函数可导

-	Softmax回归参数$(w_i, b_i$$冗余，可以消去一组

###	Softplus

$$
softplus(z) = log(exp(z)+1)
$$

> - $z = wx + b$

###	Tanh

双曲正切函数

$$
\begin{align*}
tanh(z) & = \frac {sinhz} {coshz} \\
	& = \frac {e^z - e^{-z}} {e^z + e^{-z}} \\
\end{align*}
$$

> - $z = wx + b$
> - $\frac{\partial tanh(z)}{\partial z} = (1 - tanh(z))^2$
	：非常类似普通正切函数，可以简化梯度计算

##	线性指数类

###	Elu

线性指数

$$
elu(z, \alpha) =
\left \{ \begin{array} {l}
	z & z > 0 \\
	\alpha (e^z - 1) & x \leqslant 0 \\
\end{array} \right.
$$

###	Selu

可伸缩指数线性激活：可以两个连续层之间保留输入均值、方差

-	正确初始化权重：`lecun_normal`初始化
-	输入数量足够大：`AlphaDropout`
-	选择合适的$\alpha, scale$值

$$
selu(z) = scale * elu(z, \alpha)
$$

##	线性类

###	Softsign

$$
softsign(z) = \frac z {abs(z) + 1)}
$$

###	ReLU

*Rectfied Linear Units*：修正线性单元

$$
relu(z, max) = \left \{ \begin{array} {l}
	0 & z \leq 0 \\
	z & 0 < x < max \\
	max & z \geq max \\
\end {array} \right.
$$

###	LeakyReLU

*Leaky ReLU*：带泄露的修正线性

$$
relu(z, \alpha, max) = \left \{ \begin {array} {l}
	\alpha z & z \leq 0 \\
	z & 0 < z < max \\
	max & z \geq max \\
\end {array} \right.
$$

###	PReLU

*Parametric ReLU*：参数化的修正线性

$$
prelu(z) = \left \{ \begin{array} {l}
	\alpha z & z < 0 \\
	z & z> 0 \\
\end{array} \right.
$$

> - *$\alpha$*：自学习参数（向量），需要给出权重初始化方法
	（正则化方法、约束）

###	ThreshholdReLU

带阈值的修正线性

$$
threshhold_relu(z, theta)= \left \{ \begin{array} {l}
	z & z > theta \\
	0 & otherwise \\
\end{array} \right.
$$

###	Linear

线性激活函数：不做任何改变

##	梯度消失

激活函数导数太小（$<1$），压缩**误差（梯度）**变化

