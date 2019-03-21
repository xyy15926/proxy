#	Conjugate Gradient Method

##	共轭方向

> - 设G为$n * n$阶正定对称矩阵，若$d^{(1)}, d^{(2)}$满足
	$$(d^{(1)})^T G d^{(2)} = 0$$
	则称$d^{(1)}, d^{(2)}$关于G共轭

> - 类似正交方向，若$d^{(1)},\cdots,d^{(k)}(k \leq n)$关于
	G两两共轭，则称其为G的k个共轭方向

-	特别的，$G=I$时，共轭方向就是正交方向

###	定理1

> - 设目标函数为
	$$
	f(w) = \frac 1 2 w^T w + r^T w + \sigma
	$$
	$q^{(1)}, \cdots, q^{(k)}$是$k, k \leq n$个非零正交方向
	，从任意初始点$w^{(1)}$出发，依次沿着以上正交方向做
	**精确一维搜索**，得到$w^{(1)}, \cdots, w^{(k+1)}$，
	则$w^{(k+1)}$是$f(w)$在线性流形
	$$
	\bar W_k = \{w = w^{(1)} + \sum_{i=1}^k \alpha_i q^{(i)}
		| -\infty < \alpha_i < +\infty \}
	$$
	上的唯一极小点，特别的k=n时，$w^{(n+1)}$是$f(w)$在整个
	空间上的唯一极小点

-	$\bar W_k$上的存在唯一极小点$\hat w^{(k)}$，在所有方向
	都是极小点，所以有

	$$
	<\triangledown f(\hat w^{(k)}), q^{(i)}> = 0, i=1,2,..
	$$

-	将$\hat w^{(k)}$由正交方向表示带入梯度，求出系数表达式

-	解精确搜索步长，得到$w^{(k+1)}$系数表达式

###	扩展子空间定理

> - 设目标函数为
	$$
	f(w) = \frac 1 2 x^T G x + r^T x + \sigma
	$$
	$d^{(1)}, \cdots, d^{(k)}$是$k, k \leq n$个非零正交方向
	，从任意初始点$x^{(1)}$出发，依次沿着以上正交方向做
	**精确一维搜索**，得到$x^{(1)}, \cdots, x^{(k+1)}$，
	则$x^{(k+1)}$是$f(x)$在线性流形
	$$
	\bar x_k = \{x = x^{(1)} + \sum_{i=1}^k \alpha_i d^{(i)}
		| -\infty < \alpha_i < +\infty \}
	$$
	上的唯一极小点，特别的k=n时，$x^{(n+1)}$是$f(x)$在整个
	空间上的唯一极小点

-	引进变换$w = \sqrt G x$即可证

> - 在以上假设下，有
	$$
	<\triangledown f(x^{(k+1)}), d^{(i)}> = 0, i=1,2...
	$$

##	*Conjugate Gradient Method*

共轭梯度法

###	对正定二次函数函数

$$
f(x) = \frac 1 2 x^T G x + r^T x + \sigma
$$

-	任取初始点$x^{(1)}$，若$\triangledown f(x^{(1)}) = 0$，
	停止计算，得到极小点$x^{(1)}$，否则取

	$$
	d^{(1)} = -\triangledown f(x^{(1)})
	$$

-	沿着$d^{(1)}$方向进行精确一维搜索得到$x^{(2)}$，若
	$\triangledown f(x^{(2)}) \neq 0$，令

	$$
	d^{(2)} = -\triangledown f(x^{(2)}) + \beta_1^{(2)}
		d^{(1)}
	$$

	且满足$(d^{(1)})^T G d^{(2)} = 0$，即二者共轭，可得
	
	$$
	\beta_1^{(2)} = \frac {(d^{(1)})^T G \triangledown
		f(x^{(2)})} {((d^{(1)})^T G d^{(1)})}
	$$

	-	这里$d^{(2)}$方向的构造方式是为类似构造后面$d^{(k)}$
		，得到能方便表示的系数
	-	类似于将向量组$\triangledown f(x^{(i)})$正交化

-	如此重复搜索，若$\triangledown f^(x^{i)}) \neq 0$，构造
	$x^{(k)}$处搜索方向$d^{(k)}$如下

	$$\begin{align*}
	0 & = (d^{(i)})^T G d^{(k)} \\
	& = -(d^{(i)})T G \triangledown f(x^{(k)}) +
		\sum_{j=1}^{k-1} \beta_j^{(k)} (d^{(i)})^T G d^{(j)} \\
	& = -(d^{(i)})^T G \triangledown f(x^{(k)}) +
		\beta_i^{(k)} (d^{(i)})^T G d^{(i)}
	\end{align*}$$

	可得

	$$
	\beta_i^{(k)} = \frac {(d^{(i)})^T G \triangledown
		f(x^{(k)})} {(d^{(i)})^T G d^{(i)}}
	$$

	此时$d^{(k)}$与前k-1个方向均关于G共轭，此k个方向是G的k个
	共轭方向，由扩展空间子定理，$x^{(k+1)}$是整个空间上极小

####	计算公式简化

期望简化$d^{(k)}$的计算公式

-	由扩展子空间定理推论有
	$\triangledown f(x^{(k)})^T d^{(i)} = 0, i=1,2...,k-1$
	结合以上$d^{(k)}$的构造公式，有

	$$\begin{align*}
	& \triangledown f(x^{(k)})^T \triangledown f(x^{(i)}) \\
	= & \triangledown f(x^{(k)})^T ( -d^{(i)} +
		\beta_1^{(i)} d^{(1)} + \cdots +
		\beta_{i-1}^{(i)} d^{(i-1)} ) \\
	= & 0, i=1,2,...,k-1
	\end{align*}$$

-	则有

	$$\begin{align*}
	(d^{(i)})^T G \triangledown f(x^{(k)}) & =
		\triangledown f(x^{(k)})^T G d^{(i)} \\
	& = \frac 1 {\alpha_i} \triangledown f(x^{(k)})^T
		G (x^{(i+1)} - x^{(i)}) \\
	& = \frac 1 {\alpha_i} \triangledown f(x^{(k)})^T
		(\triangledown f(x^{(i+1)}) -
		\triangledown f(x^{(i)})) \\
	& = 0, i=1,2,\cdots,k-2
	\end{align*}$$

	> - $d^{(k)} = \frac 1 {\alpha_i} x^{(i+1)} - x^{(i)}$

-	所以上述$d^{(k)}$构造公式可以简化为

	$$
	d^{(k)} = -\triangledown f(x^{(k)}) + \beta_{k-1}
		d^{(k-1)}
	$$

-	类似以上推导有

	$$\begin{align*}
	(d^{(k-1)})^T G \triangledown f(x^{(k)}) & =
		\frac 1 {\alpha_i} \triangledown f(x^{(k)})^T
		(\triangledown f(x^{(k)}) -
		\triangledown f(x^{(k-1)})) \\
	& = \frac 1 {\alpha_i} \triangledown f(x^{(k)})^T
		\triangledown f(x^{(k)}) \\
	\end{align*}$$

	$$\begin{align*}
	(d^{(k-1)})^T G d^{(k-1)} & = \frac 1 {\alpha_i}
		(d^{(k-1)})^T (\triangledown f(x^{(k)}) -
		\triangledown f(x^{(k-1)})) \\
	& = -\frac 1 {\alpha_i} (d^{(k-1)})^T
		\triangledown f(x^{(x-1)}) \\
	& = -\frac 1 {\alpha_i} (\triangledown f(x^{(k-1)}) -
		\beta_{k-2}d^{(k-2)})^T \triangledown f(x^{(x-1)}) \\
	& = -\frac 1 {\alpha_i} \triangledown f(x^{(k-1)})^T
		\triangledown f(x^{(k-1)})
	\end{align*}$$

	最终的得到简化后系数$\beta_{k-1}, k>1$的PRP公式

	$$
	\beta_{k-1} = \frac {\triangledown f(x^{(k)})^T
		(\triangledown f(x^{(k)}) -
		\triangledown f(x^{(k-1)}))}
		{\triangledown f(x^{(k-1)})^T
			\triangledown f(x^{(k-1)})}
	$$

	或FR公式

	$$
	\beta_{k-1} = \frac {\|\triangledown f(x^{(k)})\|^2}
		{\|\triangledown f(x^{(k-1)}) \|^2}
	$$

> - 以上推导虽然是根据正定二次函数得出的推导，但是仍适用于
	一般可微函数

> - $\beta _ {k-1}$给出两种计算方式，应该是考虑到目标函数
	可能不是标准正定二次函数、一维搜索数值计算不精确性

> - 将$\beta _ {k-1}$分子、分母推导到不同程度可以得到其他
	公式

-	Growder-Wolfe公式

	$$
	\beta_{k-1} = \frac {\triangledown f(x^{(k)})^T
		(\triangledown f(x^{(k)}) -
		\triangledown f(x^{(k-1)}))}
		{(d^{(k-1)})^T (\triangledown f(x^{(k)}) -
		\triangledown f(x^{(k-1)}))}
	$$

-	Dixon公式

	$$
	\beta_{k-1} = \frac {\triangledown f(x^{(k)})^T
		\triangledown f(x^{(k)})}
		{(d^{(k-1)})^T \triangledown f(x^{(k-1)})}
	$$

###	FR/PRP算法

1.	初始点$x^{(1)}$、精度要求$\epsilon$，置k=1

2.	若$\|\triangledown f(x^{(k)}) \| \leq \epsilon$，停止
	计算，得到解$x^{(k)}$，否则置

	$$
	d^{(k)} = -\triangledown f(x^{(k)}) + \beta_{k-1}d^{(k-1)}
	$$

	其中$\beta_{k-1}=0, k=1$，或由上述公式计算

3.	一维搜索，求解一维问题

	$$
	\arg\min_{\alpha} \phi(\alpha) = f(x^{(k)} -
		\alpha d^{(k)})
	$$

	得$\alpha_k$，置$x^{(k+1)} = x^{(k)} + \alpha_k d^{(k)}$

4.	置k=k+1，转2

> - 实际计算中，n步重新开始的FR算法优于原始FR算法
> - PRP算法中
	$\triangledown f(x^{(k)}) \approx \triangledown f(x^{(k-1)})$
	时，有$\beta_{k-1} \approx 0$，即
	$d^{(k)} \approx -\triangledown f(x^{(k)})$，自动重新开始
> - 试验表明，对大型问题，PRP算法优于FR算法

###	共轭方向下降性

> - 设$f(x)$具有连续一阶偏导，假设一维搜索是精确的，使用共轭
	梯度法求解无约束问题，若$\triangledown f(x^{(k)}) \neq 0$
	则搜索方向$d^{(k)}$是$x^{(k)}$处的下降方向

-	将$d^{(k)}$导入即可

###	算法二次终止性

> - 若一维搜索是精确的，则共轭梯度法具有二次终止性

-	对正定二次函数，共轭梯度法至多n步终止，否则
	-	目标函数不是正定二次函数
	-	或目标函数没有进入正定二次函数区域，

-	此时共轭没有意义，搜索方向应该重新开始，即令

	$$
	d^{(k)} = -\triangledown f(x^{(k)})
	$$

	即算法每n次重新开始一次，称为n步重新开始策略









