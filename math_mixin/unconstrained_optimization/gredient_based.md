#	基于梯度算法

##	思想

对目标函数$f(x)$在$x^{(1)}$进行展开

$$
f(x) = f(x^{(1)}) + \triangledown f(x^{(1)})(x - x^{(1)})+
	\frac 1 2 \triangledown^2 f(x^{(1)})(x - x^{(1)})^2 +
	o((x - x^{(1)})^2)
$$

> - 最速下降法：只保留一阶项，即使用线性函数近似原目标函数
> - Newton法：保留一阶、二阶项，即使用二次函数近似

-	利用近似函数求解元素问题极小值
	-	对最速下降法，线性函数无极值，需要确定步长、迭代
	-	对Newton法，二次函数有极值，直接求导算出极值、迭代

-	最速下降法
	-	只考虑一阶导，甚至说根本没有考虑拟合原目标函数

-	Newton法
	-	考虑二阶导，每步迭代还考虑了二阶导，即当前更新完毕
		后，下一步能够更好的更新（二阶导的意义），甚至说从
		后面部分可以看出，Newton法甚至考虑是全局特征，不只
		是局部性质（前提目标函数性质足够好）
	-	二次函数拟合更接近函数极值处的特征
#todo

##	最速下降算法

###	思想

-	设$x=x(t)$为最优点x以负梯度方向经过的曲线，则有

	$$\left \{ \begin{array}{l}
	& \frac {dx(t)} {dt} = -\triangledown f(x(t)) \\
	& x(t_1) = x^{(1)}
	\end{array} \right.$$

	> - $t_1, x^{(1)}$：初始时刻、初始位置

-	可以证明，$x(t)$解存在，且$t \rightarrow \infty$时，有
	$x(t) \rightarrow x^{ * }$，即得到无约束问题最优解

-	但微分方程组求解可能很麻烦，可能根本无法求解
	-	考虑将以上曲线离散化，每次前进到“不应该”前进为止
	-	然后更换方向，逐步迭代得到最优解

###	算法

> - 搜索方向最速下降方向：负梯度方向
> - 终止准则：$\triangledown f(x^{(k)})=0$

1.	取初始点$x^{(1)}$，置k=1

2.	若$\triangledown f(x^{(k)})=0$，则停止计算，得到最优解，
	否则置
	$$d^{(k)} = -\triangledown f(x^{(k)})$$
	以负梯度作为前进方向

3.	一维搜索，求解一维问题
	$$
	\arg\min_{\alpha} \phi(\alpha) =
		f(x^{(k)} + \alpha d^{(k)})
	$$
	得$\alpha_k$前进步长，置
	$$
	x^{(k+1)} = x^{(k)} + \alpha_k d^{(k)}
	$$

4.	置k=k+1，转2

> - 最速下降算法不具有二次终止性

##	Newton法

###	思想

-	若$x^{ * }$是无约束问题局部解，则有

	$$\triangledown f(x^{ * }) = 0$$

	可求解此问题，得到无约束问题最优解

-	原始问题是非线性，考虑求解其线性逼近，在初始点$x^{(1)}$
	处泰勒展开

	$$
	\triangledown f(x) \approx \triangledown f(x^{(1)})
		+ \triangledown^2 f(x^{(1)})(x - x^{(1)})
	$$

	解得

	$$
	x^{(2)} = x^{(1)} - (\triangledown^2 f(x^{(1)}))^{-1}
		\triangledown f(x^{(1)})
	$$

	作为$x^{ * }$的第二次近似

-	不断迭代，得到如下序列

	$$
	x^{(k+1)} = x^{(k)} + d^{(k)}
	$$

	> - $d^{(k)}$：Newton方向，是满足以下方程组解
		$$
		\triangledown^2 f(x^{(k)}) d = -\triangledown
			f(x^{(k)}
		$$

###	算法

1.	初始点$x^{(1)}$、精度要求$\epsilon$，置k=1

2.	若$\|\triangledown f(x^{(k)})\| \leq \epsilon$，停止计算
	，得到最优解$x^{(k)}$，否则求解

	$$
	\triangledown^2 f(x^{(k)}) d = -\triangledown
		f(x^{(k)}
	$$

	得到$d^{(k)}$

3.	置

	$$x^{(k+1)} = x^{(k)} + d^{(k)}, k = k+1$$

	转2

###	特点

-	优点
	-	产生点列$\{x^{k}\}$若收敛，则具有二阶收敛速率
	-	具有二次终止性，事实上对正定二次函数，一步即可收敛

-	缺点
	-	可能会在某步迭代时目标函数值上升
	-	当初始点$x^{(1)}$距离最优解$x^{ * }$时，产生的点列
		可能不收敛，或者收敛到鞍点
	-	需要计算Hesse矩阵
		-	计算量大
		-	Hesse矩阵可能不可逆，算法终止
		-	Hesse矩阵不正定，Newdon方向可能不是下降方向

##	阻尼/修正Newton法

-	克服Newton法目标函数值上升的缺点
-	一定程度上克服点列可能不收敛缺点

###	算法

1.	初始点$x^{(1)}$、精度要求$\epsilon$，置k=1

2.	若$\|\triangledown f(x^{(k)})\| \leq \epsilon$，停止计算
	，得到最优解$x^{(k)}$，否则求解

	$$
	\triangledown^2 f(x^{(k)}) d = -\triangledown
		f(x^{(k)}
	$$

	得到$d^{(k)}$

3.	一维搜索，求解一维问题

	$$
	\arg\min_{\alpha} \phi(\alpha) = f(x^{(k)} +
		\alpha d^{(k)})
	$$

	得到$\alpha_k$，置

	$$x^{(k+1)} = x^{(k)} + \alpha_k d^{(k)}, k = k+1$$

	转2

##	其他改进

-	针对Newton法、修正Newton法中Hesse矩阵可能不正定的改进

###	结合最速下降方向

将Newton方向和最速下降方向结合

-	设$\theta_k$是$d_N^{(k)}, -\triangledown f(x^{(k)})$之间
	夹角，显然希望$\theta < \frac \pi 2$

-	则置限制条件$\eta$，取迭代方向

	$$d^{(k)} = \left \{ \begin{array}{l}
	d_N^{(k)}, & cos\theta_k \geq \eta \\
	-\triangledown f(x^{(k)}), 其他
	\end{array} \right.$$

###	*Negative Curvature*

当Hesse矩阵非正定时，选择负曲率下降方向$d^{(k)}$（一定存在）

-	Hesse矩阵非正定时，一定存在负特征值、相应特征向量$u$

-	可以取负曲率下降方向

	$$
	d^{(k)} = -sign(u^T \triangledown f(x^{(k)})) u
	$$

> - $x^{(k)}$处负曲率方向$d^{(k)}$满足
	$$
	d^{(k)T} \triangledown^2 f(x^{(k)}) d^{(k)} < 0
	$$

###	修正Hesse矩阵

取$d^{(k)}$为以下方程的解

$$
(\triangledown^2 f(x^{(k)}) + v_k I) d =
	-\triangledown f(x^{k})
$$

> - $v_k$：大于$\triangledown^2 f(x^{(k)})$最大负特征值
	绝对值

