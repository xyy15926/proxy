#	常见分布

##	离散

##	连续

###	*P-stable Distributions*

> - *p_stable distribution*：随机变量$\sum_i v_i X_i$和变量
	$(\sum_i |v_i|^P)^{1\p} X$具有相同的分布
> > -	$v_1, v_2, \cdots, v_n$：任意实数
> > -	$X_1, X_2, \cdots, X_n$：独立同分布$D$随机变量
> > -	$X$：服从分布$D$随机变量

-	$\forall p \in (0, 2]$，稳定分布存在，但仅$p=1,2$时，
	有解析解

	-	$p=1$：柯西分布

		$$
		c(x) = \frac 1 \pi \frac 1 {1+x^2}
		$$

	-	$p=2$：高斯分布

		$$
		g(x) = \frac 1 {\sqrt {2\pi}} e^{-\frac {x^2} 2}
		$$

-	可以从$[0,1]$上均匀分布获得稳定分布
	-	但是概率分布、密度函数没有解析解

####	性质、用途

-	若向量$a$中每个元素独立从p-stable分布中抽取，则
	$\|v\|_p X = (\sum_i \|v_i\|^p)^{1/p} X$和$<a,v>$同分布

	-	用较好计算的内积估计$\|v\|_p$
	-	考虑到$a(v_1 - v_2) = av_1 - av_2$，联系了内积和点
		之间距离$\|v_1 - v_2\|_p$

##	*Exponential Family of Distributions*

单变量指数分布概率密度/分布

$$\begin{align*}
f_X(x|\theta) & = h(x) exp(\eta(\theta) T(x) - A(\theta)) \\
& h(x) g(\theta) exp(\eta(\theta) T(x)) \\
& exp(\eta(\theta) T(x) - A(\theta) + B(x))
\end{align*}$$

> - $\eta(\theta)$：*nutural parameter*，自然参数
> - $h(x)$：*underlying measure*，底层观测值
> - $T(x)$：*sufficient statistic*，随机变量X的充分统计量
> - $A(\theta)$：*log normalizer*，对数规范化

-	$\eta(\theta), T(x)$：可以是向量，其内积仍为实数

-	$\eta(\theta) = \theta$时，称分布族为*canonical*形式
	-	总是能够定义$\eta = \eta(\theta)$转为此形式

-	对数规范化$A(\theta)$使得概率密度函数满足积分为1

	$$\begin{align*}
	f(x|\theta) exp(A(\theta)) & = h(x)
		exp(\eta(\theta)T(x)) \\
	\int exp(A(\theta)) f(x|\theta) dx & =
		\int h(x) exp(\eta(\theta) T(x)) dx \\
	exp(A(\theta)) \int f(x|\theta) dx & =
		\int h(x) exp(\eta(\theta) T(x)) dx \\
	A(\theta) = ln \int h(x) exp(\eta(\theta) T(x)) dx
	\end{align*}
	$$

###	性质

-	充分统计量$T(x)$可以使用固定几个值，从大量的独立同分布
	数据中获取信息
#todo

###	*Bernoulli*分布

-	$h(x) = 1$
-	$T(x) = x$
-	$\eta = log \frac \theta {1 - \theta}$
	$A()

###	*Possion*

-	$\theta = \lambda$
-	$h(x) = \frac 1 {x!}$
-	$\eta(\theta) = ln\lambda$
-	$T(x) = x$
-	$A(\theta) = \lambda$

###	*Normal*

-	$h(x) = \frac 1 \sqrt{2\pi\sigma^2} e^{-\frac {x^2} {2\sigma^2}}$
-	$T(x) = \frac x \sigma$
-	$A(\theta) = \frac {\mu^2} {2\sigma^2}$
-	$\eta(\theta) = \frac \mu \sigma$







