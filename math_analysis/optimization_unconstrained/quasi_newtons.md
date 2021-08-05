---
title: Quasi-Newton Method/Variable Metric Method
categories:
  - Math Analysis
  - Optimization
tags:
  - Math
  - Analysis
  - Optimization
  - Unconstrained
  - Newton
date: 2019-04-23 01:32:09
updated: 2019-04-23 01:32:09
toc: true
mathjax: true
comments: true
description: Quasi-Newton Method/Variable Metric Method
---

##	综述

拟Newton法/变度量法：不需要求解Hesse矩阵，使用一阶导构造
二阶信息的近似矩阵

-	使用迭代过程中信息，创建近似矩阵$B^{(k)}$代替Hesse矩阵

-	用以下方程组替代Newton方程，其解$d^{(k)}$作为搜索方向

	$$
	B^{(k)} d = - \triangledown f(x^{(k)})
	$$

###	思想

-	考虑$\triangledown f(x)$在$x^{(k+1)}$处泰勒展开

	$$
	\triangledown f(x) \approx \triangledown f(x^{(k+1)})
		+ \triangledown^2 f(x^{(k+1)})(x - x^{(k+1)})
	$$

-	取$x = x^{(k)}$，有

	$$\begin{align*}
	\triangledown f(x^{(k+1)}) - \triangledown f(x^{(k)})
		& \approx \triangledown^2 f(x^{(x+1)})
		(x^{(k+1) } - x^{(k)}) \\
	\triangledown^2 f(x^{k+1}) s^{(k)} & \approx y^{(k)}
	\end{align*}$$

	> - $s^{(k)} = x^{(k+1)} - x^{(k)}$
	> - $y^{(k)} = \triangledown f(x^{(k+1)}) - \triangledown f(x^{(k)})$

-	要求$B^{(k)}$近似$\triangledown^2 f(x^{(k)})$，带入并将
	$\approx$改为$=$，得到拟Newton方程

	$$
	B^{(k+1)} s^{(k)} = y^{(k)}
	$$

	并假设$B^{(k)}$对称

-	拟Newton方程不能唯一确定$B^{(k+1)}$，需要附加条件，自然
	的想法就是$B^{(k+1)}$可由$B^{(k)}$修正得到，即

	$$
	B^{(k+1)} = B^{(k)} + \Delta B^{(k)}
	$$

	且修正项$\Delta B^{(k)}$具有“简单形式”

##	Hesse矩阵修正

###	对称秩1修正

认为简单指矩阵秩小：即认为$\Delta B^{(k)}$秩为最小值1

-	设$\Delta B^{(k)} = u v^T$，带入有

	$$\begin{align*}
	y^{(k)} & = B^{(k+1)} s^{(k)} \\
	& = B^{(k)} s^{(k)} + (v^T s^{(k)}) u \\
	y^{(k)} - B^{(k)} s^{(k)} & = (v^T s^{(k)}) u
	\end{align*}$$

	-	这里有的书会设$\Delta B^{(k)} = \alpha u v^T$，
		其实对向量没有必要
	-	$v^T s^{(k)}$是数，所以$u$必然与共线，同理也没有必要
		考虑系数，直接取相等即可
	-	而且系数不会影响最终结果

-	**可取**$u = y^{(k)} - B^{(k)} s{(k)}$，取$v$满足
	$v^T s^{(k)}  = 1$

-	由$B^{(k)}$的对称性、并希望$B^{(k+1)}$保持对称，需要
	$u, v$共线，则有

	$$\begin{align*}
	v & = \lambda u = \lambda (y^{(k)} - B^{(k)} s^{(k)}) \\
	1 & = \lambda (y^{(k)} - B^{(k)} s^{(k)})^T s^{(k)}
	\end{align*}$$

-	得到$B^{(k)}$的对称秩1修正公式

	$$
	B^{(k+1)} = B^{(k)} + \frac {(y^{(k) - B^{(k)} s^{(k)}})
		(y^{(k)} - B^{(k)} s^{(k)})^T}
		{(y^{(k)} - B^{(k)} s^{(k)})^T s^{(k)}}
	$$

####	算法

1.	初始点$x^{(1)}$、初始矩阵$B^{(1)} = I$、精度要求
	$\epsilon$、置$k=1$

2.	若$\|\triangledown f(x^{(k)})\| \leq \epsilon$，停止计算
	，得到解$x^{(k)}$，否则求解以下方程得到$d^{(k)}$

	$$
	B^{(k)} d = -\triangle f(x^{(k)})
	$$

3.	一维搜索，求解

	$$
	\arg\min_{\alpha} \phi(\alpha)=f(x^{(k)} + \alpha d^{(k)})
	$$

	得到$\alpha_k$，置$x^{(k+1)}=x^{(k)} + \alpha_k d^{(k)}$

4.	修正$B^{(k)}$

	$$\begin{align*}
	s^{(k)} & = x^{(k+1)} - x^{(k)} \\
	y^{(k)} & = \triangledown f(x^{(k+1)}) -
		\triangledown f(x^{(k)}) \\
	B^{(k+1)} & = B^{(k)} + \frac {(y^{(k) - B^{(k)} s^{(k)}})
		(y^{(k)} - B^{(k)} s^{(k)})^T}
		{(y^{(k)} - B^{(k)} s^{(k)})^T s^{(k)}}
	\end{align*}$$

5.	置$k = k+1$，转2

####	特点

-	缺点

	-	要求$(y^{(k)} - B^{(k)} s^{(k)})^T s^{(k)} \neq 0$，
		否则无法继续计算

	-	不能保证正定性传递，只有
		$(y^{(k)} - B^{(k)} s^{(k)})^T s^{(k)} > 0$才能保证
		$B^{(k+1)}$也正定

	-	即使$(y^{(k)} - B^{(k)} s^{(k)})^T s^{(k)} > 0$，
		也可能很小，容易产生较大的舍入误差

###	对称秩2修正

-	为克服秩1修正公式缺点，考虑$\Delta B^{(k)}$秩为2，设

	$$
	\Delta B^{(k)} = u^{(1)} (v^{(1)})^T
		+ u^{(2)} (v^{(2)})^T
	$$

-	带入拟Newton方程有

	$$
	B^{(k)} s^{(k)} + ((v^{(1)})^T s^{(k)}) u^{(1)} +
		((v^{(2)})^T s^{(k)}) u^{(2)} = y^{(k)}
	$$

-	类似的取

	$$\left \{ \begin{array}{l}
	u^{(1)} = y^{(k)} \\
	(v^{(1)})^T s^{(k)} = 1
	\end{array} \right.$$

	$$\left \{ \begin{array}{l}
	u^{(2)} = -B^{(k)} s^{(k)} \\
	(v^{(2)})^T s^{(k)} = 1
	\end{array} \right.$$

-	同秩1公式保持对称性推导，得到对称秩2修正公式/BFGS公式

	$$
	B^{(k+1)} = B^{(k)} - \frac {B^{(k)} s^{(k)}
		(s^{(k)})^T B^{(k)}} {(s^{(k)})^T B^{(k)} s^{(k)}}
		+ \frac {y^{(k)} (y^{(k)})^T} {(y^{(k)})^T s^{(k)}}
	$$

###	BFGS算法

类似同秩1修正算法，仅第4步使用对称秩2修正公式

##	Hesse逆修正

###	对称秩2修正

-	考虑直接构造近似于$(\triangledown^2 f(x^{(k)}))^{-1}$的
	矩阵$H^{(k)}$
	
-	这样无需求解线性方程组，直接计算
	$$
	d^{(k)} = -H^{(k)} \triangledown f(x^{(k)})
	$$

-	相应拟Newton方程为
	$$
	H^{(k+1)} y^{(k)} = s^{(k)}
	$$

-	可得$H^{(k)}$的对称秩1修正公式

	$$
	H^{(k+1)} = H^{(k)} + \frac {(s^{(k)} - H^{(k)} y^{(k)})
		(s^{(k)} - H^{(k)} y^{(k)})T}
		{(s^{(k)} - H^{(k)} y^{(k)})^T y^{(k)}}
	$$

-	可得$H^{(k)}$的对称秩2修正公式/DFP公式

	$$
	H^{(k+1)} = H^{(k)} - \frac {H^{(k)} y^{(k)} (y^{(k)})^T
		H^{(k)}} {(y^{(k)})^T H^{(k)} y^{(k)}} +
		\frac {s^{(k)} (s^{(k)})^T} {(s^{(k)})^T y^{(k)}}
	$$

####	DFP算法

类似BFGS算法，只是

-	使用$H^{(k)}$计算更新方向
-	使用$H^{(k)}$的对称秩2修正公式修正

> - 对正定二次函数，BFGS算法和DFP算法效果相同
> - 对一般可微（非正定二次函数），一般认为BFGS算法在收敛性质
	、数值计算方面均由于DFP算法

###	Hesse逆的BFGS算法

-	考虑

	$$\begin{align*}
	B^{(k+1)} & = B^{(k)} + u^{(1)} (v^{(1)})^T +
		u^{(2)} (v^{(2)})^T \\
	H^{(k+1)} & = (B^{(k+1)})^{-1} \\
	& = (B^{(k)} + u^{(1)} (v^{(1)})^T + u^{(2)}
		(v^{(2)})^T)^{-1} \\
	\end{align*}$$

-	两次利用*Sherman-Morrison*公式，可得

	$$
	H^{(k+1)} = (I - \frac {s^{(k)} (y^{(k)})^T} 
		{(y^{(k)})^T s^{(k)}})
		H^{(k)}
		(I - \frac {s^{(k)} (y^{(k)})^T}
			{(y^{(k)})^T s^{(k)}})^T
		+ \frac {s^{(k)} (s^{(k)})^T} {(y^{(k)})^T s^{(k)}}
	$$

#todo

-	还可以进一步展开

	$$
	H^{(k+1)} = H^{(k)} + (\frac 1 {(s^{(k)})^T y^{(k)}} +
		\frac {(y^{(k)})^T H^{(k)} y^{(k)}}
		{((s^{(k)})^T y^{(k)})^2}) s^{(k)} (s^{(k)})^T
		- \frac 1 {(s^{(k)})^T y^{(k)}}
		(H^{(k)} y^{(k)} (s^{(k)})^T +
		s^{(k)} (y^{(k)})^T H^{(k)})
	$$

##	变度量法的基本性质

###	算法的下降性

####	定理1

> - 设$B^{(k)}$（$H^{(k)}$）是正定对称矩阵，且有
	$(s^{(k)})^T y^{(k)} > 0$，则由BFGS（DFS）公式构造的
	$B^{(k+1)}$（$H^{(k+1)}$）是正定对称的

-	考虑$B^{(k)}$对称正定，有
	$B^{(k)} = (B^{(k)})^{1/2} (B^{(k)})^{1/2}$

-	带入利用柯西不等式即可证

> - 中间插入正定矩阵的向量内积不等式也称为广义柯西不等式

####	定理2

> - 若$d^{(k)}$v是下降方向，且**一维搜索是精确的**，设
	$B^{(k)}$（$H^{(k)}$）是正定对称矩阵，则有BFGS（DFP）
	公式构造的$B^{(k+1)}$（$H^{(k+1)}$）是正定对称的

-	精确一维搜索$(d^{(k)})^T \triangledown f(x^{(k+1)}) = 0$
-	则有$(s^{(k)})^T y^{(k)} > 0$

####	定理3

> - 若用BFGS算法（DFP算法）求解无约束问题，设初始矩阵
	$B^{(1)}$（$H^{(1)}$）是正定对称矩阵，且一维搜索是精确的
	，若$\triangledown f(x^{(k)}) \neq 0$，则产生搜索方向
	$d^{(k)}$是下降方向

-	结合上2个结论，数学归纳法即可

####	总结

-	若每步迭代一维搜索精确，或满足$(s^{(k)})^T y^{(k)} > 0$
	-	停止在某一稳定点
	-	或产生严格递减的序列$\{f(x^{(k)})\}$

-	若目标函数满足一定条件我，可以证明变度量法产生的点列
	$\{x^{(k)}\}$收敛到极小点，且收敛速率超线性

###	搜索方向共轭性

> - 用变度量法BFGS（DFP）算法求解正定二次函数

	$$
	min f(x) = \frac 1 2 x^T G x + r^T x + \sigma
	$$

	若一维搜索是精确的，假设已经进行了m次迭代，则

> - 搜索方向$d^{(1)}, \cdots, d^{(m)}$是m个非零的G共轭方向

> - 对于$j = 1, 2, \cdots, m$，有

	$$
	B^{(m+1)} s^{(j)} = y^{(j)}
	(H^{(m+1)} y^{(j)} = s^{(j)})
	$$

	且$m = n$时有吧

	$$
	B^{(n+1)} = G(H^{(n+1)} = G^{-1})
	$$

###	变度量法二次终止

> - 若一维搜索是精确的，则变度量法（BFGS、DFP）具有二次终止

-	若$\triangle f(x^{(k)}) = 0, k \leq n$，则得到最优解
	$x^{(k)}$

-	否则得到的搜索方向是共轭的，由扩展空间子定理，
	$x^{(n+1)}$是最优解






