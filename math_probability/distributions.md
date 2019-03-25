#	常见分布

##	离散

##	连续

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







