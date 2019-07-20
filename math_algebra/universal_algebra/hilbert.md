#	Hilbert空间

##	*Reproducing Kernel Hilbert Space*

假设$K(x,z)$是定义在$\mathcal{X * X}$上的**对称函数**，并且
对任意$x_1, x_2, \cdots, x_m \in \mathcal{X}$，$K(x,z)$关于
其的Gram矩阵半正定，则可以根据函数$K(x,z)$构成一个
*Hilbert space*

###	构造步骤

####	定义映射构成向量空间

-	定义映射

	$$\phi: x \rightarrow K(·, x)$$

-	根据此映射，对任意
	$x_i \in \mathcal{X}, \alpha_i \in R, i = 1,2,...,m$定义
	线性组合

	$$
	f(·) = \sum_{i=1}^m \alpha_i K(·, x_i)
	$$

-	由以上线性组合为元素的集合$S$对加法、数乘运算是封闭的，
	所以$S$构成一个向量空间

####	定义内积构成内积空间

在$S$上定义运算$*$：对任意$f, g \in S$

$$\begin{align*}
f(·) = \sum_{i=1}^m \alpha_i K(·, x_i) \\
g(·) = \sum_{j=1}^n \beta_j K(·, z_j)
\end{align*}$$

定义运算$*$

$$
f * g = \sum_{i=1}^m \sum_{j=1}^n \alpha_i \beta_j
	K(x_i, z_j)
$$

为证明运算$*$是空间$S$的内积，需要证明

1.	$(cf) * g = c(f * g), c \in R$
2.	$(f + g) * h = f * h + g * h, h \in S$
3.	$f * g = g * f$
4.	$f * f \geq 0, f * f = 0 \Leftrightarrow f = 0$

-	其中1-3由$S$元素定义、$K(x,z)$对称性容易得到

-	由$*$运算规则可得

	$$
	f * f = \sum_{i,j=1}^m \alpha_i \alpha_j K(x_i, x_j)
	$$

	由Gram矩阵非负可知上式右端非负，即$f * f \geq 0$

-	为证明$f * f \Leftarrow f = 0$

	-	首先证明

		$$
		|f * g|^2 \leq (f * f)(g * g)
		$$

		-	设$f, g \in S$，则有$f + \lambda g \in S$，则有

			$$\begin{align*}
			(f + \lambda g) * (f + \lambda g) & \geq 0 \\
			f*f + 2\lambda (f * g) + \lambda^2 (g*g) & \geq 0
			\end{align*}$$

		-	则上述关于$\lambda$的判别式非负，即

			$$
			(f*g)^2 - (f*f)(g*g) \leq 0
			$$

	-	$\forall x \in \mathcal{x}$，有

		$$
		K(·, x) * f = \sum_{i=1}^m \alpha_i K(x, x_i) = f(x)
		$$

		则有

		$$
		|f(x)|^2 = |K(·, x) * f|^2
		$$

	-	又

		$$
		|K(·, x) * f|^2 \leq (K(·, x) * K(·, x))(f * f) =
			K(x, x)(f*f)
		$$

		则有

		$$
		|f(x)|^2 \leq K(x, x) (f * f)
		$$

		即$f * f = 0$时，对任意$x$都有$|f(x)| = 0$

因为$*$为向量空间$S$的内积，可以继续用$·$表示

$$
f·g = \sum_{i=1}^m \sum_{j=1}^n \alpha_i \alpha_j K(x_i,z_J)
$$

####	完备化构成Hilbert空间

-	根据内积定义可以得到范数

	$$
	\|f\| = \sqrt {f · f}
	$$

	所以$S$是一个赋范向量空间，根据泛函分析理论，对于不完备
	的赋范空间$S$，一定可以使之完备化得到希尔伯特空间
	$\mathcal{H}$

-	此希尔波特空间$\mathcal{H}$，称为
	*reproducing kernel Hilber Space$，因为核$K$具有再生性

	$$\begin{align*}
	K(·, x) · f & = f(x) \\
	K(·, x) · K(·, Z) & = K(x, z)
	\end{align*}
	$$

###	*Positive Definite Kernel Function*

> - 设$K: \mathcal{X * X} \leftarrow R$是对称函数，则
	$K(x,z)$为正定核函数的充要条件是
	$\forall x_i \in \mathcal{X}, i=1,2,...,m$，$K(x,z)$对应
	的Gram矩阵
	$$
	K = [K(x_i, x_j)]_{m*m}
	$$
	是半正定矩阵

必要性

-	由于$K(x,z)$是$\mathcal{X * X}$上的正定核，所以存在从
	$\mathcal{X}$到Hilbert空间$\mathcal{H}$的映射，使得

	$$
	K(x,z) = \phi(x) \phi(z)
	$$

-	则对任意$x_1, x_2, \cdots, x_m$，构造$K(x,z)$关于其的
	Gram矩阵

	$$
	[K_{ij}]_{m*m} = [K(x_i, x_i)]_{m*m}
	$$

-	对任意$c_1, c_2, \cdots, c_m \in R$，有

	$$\begin{align*}
	\sum_{i,j=1}^m c_i c_j K(x_i, x_j) & = \sum_{i,j=1}^m
		c_i c_j (\phi(x_i) \phi(x_j)) \\
	& = (\sum_i c_i \phi(x_i))(\sum_j c_j \phi(x_j)) \\
	& = \| \sum_i c_i \phi(x_i) \|^2 \geq 0
	\end{align*}$$

	所以$K(x,z)$关于$x_1, x_2, \cdots, x_m$的Gram矩阵半正定

充分性

-	对给定的$K(x,z)$，可以构造从$\mathcal{x}$到某个希尔伯特
	空间的映射

	$$
	\phi: x \leftarrow K(·, x)
	$$

-	且有

	$$
	K(x,z) = \phi(x) · \phi(z)
	$$

	所以$K(x,z)$是$\mathcal{X * X}$上的核函数

