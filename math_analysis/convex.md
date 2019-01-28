#	凸分析

###	Notations and Terminology

考虑非空凸集$C \subseteq R^N$

####	*Distance*

点$x \in R^N$到$C$的距离为

$D_C(x) = \min_{y in C} \|x-y\|_2$ 

####	*Project*

如果C是闭凸集，那么点$x \in R^N$在$C$上投影为$P_Cx$

$$
P_Cx \in C, D_C(x) = \|x - P_Cx\|_2
$$

####	*Indicator Function*

C的示性函数为

$$
l_C(x) = \left \{ \begin{array}{c}
	0 & if x \in C \\
	+\infty & if x \notin C
\end{array}{c} \right.
$$




