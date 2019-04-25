#	Uncharted

##	Unregularized Least Squares Learning Problem

###	Problem

$$
w_T = \frac \gamma n \sum_{i=0}^{T-1} (I - \frac \gamma n
	{\hat X}^T \hat X)^i {\hat X}^T \hat Y
$$

> - $\gamma$：被引入保证
	$\|I - \frac \gamma n {\hat X}^T \hat X\| < 1$

###	策略

$$
\min_w I_s(w) = \frac 1 {2n} \|\hat X w - \hat Y\|^2
$$

###	算法

$$
w_0 = 0 \\
w_{t+1} = (I - \frac \gamma n {\hat X}^T \hat X)w_t +
	\frac \gamma n {\hat X}^T \hat Y
$$

> - 将$w_{t+1}$带入$I_s(w)$即可证明每次迭代$I_s(w)$减小

##	投影算法

###	*Projection Operator*

投影算子：线性变换$P$满足从向量空间到其自身$P^2=P$

-	即当线性变换$P$对任何值应用两次，将得出一样结果，就像
	只应用一次，保持像不变

