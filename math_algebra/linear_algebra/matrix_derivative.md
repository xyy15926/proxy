#	*Matrix Derivative*/*Matrix Differential*

##	矩阵求导/矩阵微分

矩阵求导：在矩阵空间的多元微积分

###	*Layout Conventions*

> - *numerator layout*：分子布局，微分分子的维数决定微分结果
	的高维度结构（行优先，如：微分矩阵行数等于分子维数）
> - *denominator layout*：分母布局，微分分母的维数为微分结果
	的高维度结构（行优先）

-	两种布局方式相差转置
-	与微分分子、分母为行、或列向量无关
-	以上布局模式是指**简单单因子微分**时布局模式，复合多因子
	应使用维度分析考虑
	（若严格按照计算规则，结果应该满足布局）

![matrix_derivative_results](imgs/matrix_derivative_results.png)

> - 数分中Jaccobi行列式采用分子布局，以下默认为分子布局

##	关于标量导数

###	标量对标量

标量$y$对标量$x$求导：$\frac {\partial y} {\partial x}$

![matrix_derivative_scalar_by_scalar_vector_involved](imgs/matrix_derivative_scalar_by_scalar_vector_involved.png)

![matrix_derivative_scalar_by_scalar_matrix_involved](imgs/matrix_derivative_scalar_by_scalar_matrix_involved.png)

###	向量对标量

向量$Y$关于标量$x$求导（$Y$为行、列向量均如此）

![matrix_derivative_vector_by_scalar](imgs/matrix_derivative_vector_by_scalar.png)

$$
\frac {\partial Y} {\partial x} = \begin{bmatrix}
	\frac {\partial y_1} {\partial x} \\
	\frac {\partial y_2} {\partial x} \\
	\vdots \\
	\frac {\partial y_n} {\partial x}
\end{bmatrix}
$$

###	矩阵对标量

矩阵$Y$关于标量$x$求导

![matrix_derivative_matrix_by_scalar](imgs/matrix_derivative_matrix_by_scalar.png)

$$
\frac {\partial Y} {\partial x} = \begin{bmatrix}
	\frac {\partial y_{11}} {\partial x} & \frac
		{\partial y_{12}} {\partial x} & \cdots & \frac
		{\partial y_{1n}} {\partial x} \\
	\frac {\partial y_{21}} {\partial x} & \frac
		{\partial y_{22}} {\partial x} & \cdots & \frac
		{\partial y_{2n}} {\partial x} \\
	\vdots & \vdots & \ddots & \vdots \\
	\frac {\partial y_{n1}} {\partial x} & \frac
		{\partial y_{n2}} {\partial x} & \cdots & \frac
		{\partial y_{nn}} {\partial x} \\
\end{bmatrix}
$$

##	关于向量导数

###	标量对向量

标量$y$关于向量$X$求导

![matrix_derivative_scalar_by_vector](imgs/matrix_derivative_scalar_by_vector_1.png)
![matrix_derivative_scalar_by_vector](imgs/matrix_derivative_scalar_by_vector_2.png)

$$
\frac {\partial y} {\partial X} = [\frac {\partial y} 
	{\partial x_1}, \frac {\partial y} {\partial x_1},
	\cdots, \frac {\partial y} {\partial x_n}]
$$

###	向量对向量

向量$Y$关于向量$X$求导

![matrix_derivative_vector_by_vector](imgs/matrix_derivative_vector_by_vector.png)

$$
\frac {\partial Y} {\partial X} = \begin{bmatrix}
	\frac {\partial y_1} {\partial x_1} & \frac
		{\partial y_1} {\partial x_2} & \cdots & \frac
		{\partial y_1} {\partial x_n} \\
	\frac {\partial y_2} {\partial x_1} & \frac
		{\partial y_2} {\partial x_2} & \cdots & \frac
		{\partial y_2} {\partial x_n} \\
	\vdots & \vdots & \ddots & \vdots \\
	\frac {\partial y_m} {\partial x_1} & \frac
		{\partial y_m} {\partial x_2} & \cdots & \frac
		{\partial y_m} {\partial x_n}
\end{bmatrix}
$$

> - $Y$、$X$为行、列向量均如此

##	关于矩阵导数

###	标量对矩阵求导

![matrix_derivative_scalar_by_matrix_1](imgs/matrix_derivative_scalar_by_matrix_1.png)
![matrix_derivative_scalar_by_matrix_2](imgs/matrix_derivative_scalar_by_matrix_2.png)
![matrix_derivative_scalar_by_matrix_3](imgs/matrix_derivative_scalar_by_matrix_3.png)
![matrix_derivative_scalar_by_matrix_4](imgs/matrix_derivative_scalar_by_matrix_4.png)

##	微分

###	微分形式

![matrix_differential](imgs/matrix_differential.png)

###	导数、微分转换

![matrix_derivative_differential_conversion](imgs/matrix_derivative_differential_conversion.png)

##	维度分析

维度分析：对求导结果的维度进行分析，得到矩阵微分结果

-	维度一般化：将向量、矩阵**维度置不同值**，便于考虑转置
-	拆分有关因子：利用**求导乘法公式**分别考虑因子微分结果
-	变换微分因子、剩余因子（可能有左右两组）满足矩阵运算维度
	要求
	-	微分因子**按布局模式考虑维度、不转置**
	-	有时维度一般化也无法唯一确定剩余因子形式，考虑行、列
		內积对应关系

###	例

-	考虑$\frac {\partial x^T A x} {\partial x}$，其中
	$A \in R^{n*n}, x \in R^n$

-	维度一般化：$\frac {\partial u^T A v} {\partial x}$，
	其中$A \in R^{a * b}, x \in R^n$

-	拆分有关因子

	$$\begin{align*}
	\frac {\partial (u^T A) v} {\partial x} & = u^T A \frac
		{\partial v} {\partial x} \\
	\frac {\partial u^T (A v)} {\partial x} & = v^T A^T \frac
		{\partial u} {\partial x}
	\end{align*}$$

-	则有

	$$
	\frac {\partial x^T A x} {\partial x} = x^T (A^T + A)
	$$

