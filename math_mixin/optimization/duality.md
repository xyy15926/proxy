#	对偶理论

##	*Langrangian Duality*

拉格朗日对偶

-	考虑优化问题：找到$f(x)$满足约束的最好下界

	$$
	z^{*} = \min_{x} f(x) \\
	\begin{align*}
	s.t. \quad & g_i(x) \leq 0, i=1,2,\cdots,m \\
		& x \in X
	\end{align*}
	$$

-	考虑方程组

	$$
	\left \{ \begin{array}{l}
	f(x) < v \\
	g_i(x) \leq 0, i=1,2,\cdots,m
	\end{array} \right.
	$$

	-	方程组无解时，$v$是优化问题的一个下界

	-	若方程组有解，则可以推出

		$$
		\forall \lambda \geq 0,
		f(x) + \sum_{i=1}^m \lambda_ig_i(x) < v
		$$

	-	根据逆否命题，方程组无解的**充分条件**是

		$$
		\exists \lambda \geq 0,
		\min_{x} f(x) + \sum _{i=1}^m \lambda_ig_i(x) \geq v
		$$

-	由此方法推出的最好下界，即拉格朗日对偶问题

	$$
	v^{*} = \max_{\lambda \geq 0} \min_{x} f(x) +
		\sum_{i=1}^m \lambda_ig_i(x)
	$$

###	说明

-	拉格朗日对偶对实数域上的优化问题都存在，对目标函数、
	约束函数都没有要求

-	强对偶定理：$v^{*} = z^{*}$，需要$f,g$满足特定条件才成立

	-	线性规划
	-	半正定规划
	-	凸优化

	-	应该是中间充分条件那步所以，强对偶定理不一定成立
	-	#todo

-	弱对偶定理：$v^{*} \leq z^{*}$，永远成立（以上即可证）

	-	通过弱对偶定理，可以得到原问题的一个下界
	-	对求解原问题有帮助，比如：分支界限法中快速求下界

-	对偶问题相关算法往往原问题算法在实际应用中往往更加有效

	-	*dual-simplex*
	-	*primal-dual interior point method*
	-	*augmented Lagrangian Method*







