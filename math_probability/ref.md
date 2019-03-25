#	参考

###	Azuma-Hoeffding Inequality

设${X_i:i=0,1,2,\cdots}$是鞅差序列，且
$|X_k - X_{k-1}| < c_k$，则

$$
\begin{align*}
super-martingale:
P(X_N - X_0 \geq t) \leq exp \left ( \frac {-t^2}
	{2\sum^N_{k=1} c_k^2} \right ) \\
sub-martingale:
P(X_N - X_0 \leq -t) \leq exp \left ( \frac {-t^2}
	{2\sum^N_{k=1} c_k^2} \right ) \\
martingale:
P(|X_N - X_0| \geq t) \leq exp \left ( \frac {-t^2}
	{2\sum^N_{k=1} c_k^2} \right )
\end{align*}
$$

###	Hoeffding不等式

设$S_n = \sum_{i=1}^N X_i$是随机变量$X_1, X_2, \cdots, X_N$
之和，$X_i \in [a_i, b_i]$，则对于任意$t>0$，以下不等式成立

$$
\begin{align*}
P(S_n - ES_n \geqslant t) & \leqslant exp \left (
	\frac {-2t^2} {\sum_{i=1}^n (b_i - a_i)^2} \right ) \\
P(ES_n - S_n \geqslant t) & \leqslant exp \left (
	\frac {-2t^2} {\sum_{i=1}^n (b_i - a_i)^2} \right )  \\
\end{align*}
$$

-	这两个不等式不能用绝对值合并，分别描述不同分段概率

###	Bretagnolle-Huber-Carol Inequility

${X_i: i=1,2,\cdots,N} i.i.d. M(p1, p_2, \cdots, p_k)$
服从类别为k的多项分布

$$
p{\sum_{i=1}^k |N_i - Np_i| \geq \epsilon} \leq
	2^k exp \left ( \frac {- n\epsilon^2} 2  \right )
$$

> - $N_i$：第i类实际个数

###	条件概率分布似然函数

-	离散随机变量$(X,Y)$的条件概率分布似然函数

	$$\begin{align*}
	O_{\tilde P}(P_w) & = \prod_{i=1}^N P(y_i|x_i) \\
	& = \prod_{x,y} P(y|x)^{N * \tilde P(x,y)}
	\end{align*}$$

	> - $N$：样本数量
	> - $\tilde P(x,y)$：经验分布

-	对数似然函数为

	$$\begin{align*}
	L_{\tilde P}(P_w) & = log \prod_{x,y}
		P(y|x)^{N * \tilde P(x,y)} \\
	& = \sum_{x,y} N * \tilde P(x,y) log P(y|x) \\
	\end{align*}$$

	系数$N$可以省略，则有

	$$\begin{align*}
	L_{\tilde P}(P_w) & = log \prod_{x,y}
		P(y|x)^{\tilde P(x,y)} \\
	& = \sum_{x,y} \tilde P(x,y) log P(y|x)
	\end{align*}$$




