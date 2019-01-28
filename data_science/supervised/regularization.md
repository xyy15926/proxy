##	模型选择算法

###	最优子集回归

-	可以得到稀疏的模型
-	但搜索空间离散，可变性大，稳定性差![Breiman_1996](Breiman)

##	正则化方法

约束参数解空间

###	Ridge Regression

Hoerl and Kennard, 1988

$$
\min_{\beta \in R^n} \left\{ ||y - X\beta||_2^2 +
	\lambda ||\beta||_2^2 \right\}
$$

-	在L2范数约束下最小化残差平方
-	作为连续收缩方法
	-	通过bias-variance trade-off，ridge拥有较普通最小二乘
		更好的预测表现
	-	倾向于保留所有特征，无法产生疏系数模型

###	LASSO

$$
\min_{\beta \in R^n} \left\{ ||y - X\beta||_2^2 +
	\lambda||\beta||_1 \right\}
$$

能够选择部分特征，产生疏系数模型

-	p > n时，即使所有特征都有用，LASSO也只能从中挑选n个
-	如果存在相关性非常高的特征，LASSO倾向于只从该组中选择
	一个特征，而且是随便挑选的
	-	极端条件下，两个完全相同的特征函数，严格凸的罚函数
		（如Ridge）可以保证最优解在两个特征的系数相等，而
		LASSO的最优解甚至不唯一

####	解法

-	LARS


###	Elastic Net

####	Naive Elastic Net

$$
\begin{align*}
& \min_{\beta \in R^n} \left\{ ||y - X\beta||_2^2 +
	\lambda_1||\beta||_1 + \lambda_2||\beta||_2^2 \right\} \\

& \Rightarrow
\min_{\beta^* \in R^p} \left\{ ||y - X^*\beta^*||_2^2 +
	\lambda^*||\beta^*||_1 \right\} \\

& where: \\
& y^* = \begin{pmatrix}
  		y \\ \vec 0_p
  	\end{pmatrix}	\\
& X^* = \frac 1 {\sqrt {1+\lambda^2}}
  \begin{pmatrix}
  	X \\ \sqrt {\lambda_2} I_p
  \end{pmatrix} \\
& \beta^* = \sqrt {1+\lambda_2} \beta \\
& \lambda^* = \frac {\lambda_1} {1+\lambda_2} \\
\end{align*}
$$

-	弹性网在Lasso的基础上添加系数的二阶范数
	-	能同时做变量选择和连续收缩
	-	并且可以选择一组变量

-	传统的估计方法通过二阶段估计找到参数
	-	首先设置ridge系数$\lambda_2$求出待估参数$\beta$，
		然后做lasso的收缩
	-	这种方法有两次收缩，会导致估计偏差过大，估计不准

-	弹性网可以变换为LASSO，因而lasso的求解方法都可以用于
	elastic net


![elastic_net](http://www.stat.purdue.edu/~tlzhang/mathstat/ElasticNet.pdf)

####	To Lasso






