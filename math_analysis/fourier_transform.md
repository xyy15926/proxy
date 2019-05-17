#	Fourier Transformation

##	傅里叶变换

###	*Discrete Fourier Transformation*

*DFT*：归一化二维离散傅里叶变换

$$\begin{align*}
F(u,v) & = \frac 1 {\sqrt{NM}} \sum_{x=0}^{N-1}
	\sum_{y=0}^{M-1} f(x,y) e^{-\frac {2\pi i} N ux}
	e^{-\frac {2\pi i} M vy} \\
f(x,y) & = \frac 1 {\sqrt{NM}} \sum_{u=0}^{N-1}
	\sum_{v=0}^{M-1} F(u,v) e^{\frac {2\pi i} N ux}
	e^{\frac {2\pi i} M vy} \\
\end{align*}$$

###	*Discrete Consine Transformation*

余弦变换

> - 在给定区间为满足狄利克雷条件的连续实对称函数，可以展开为
	仅含余弦项的傅里叶级数

-	对于定义在正实数域上的函数，可以通过偶延拓、或奇延拓满足
	上述条件

###	离散余弦变换

-	$(x,y) or (u,v) = (0,0)$时

	$$\begin{align*}
	F(u,v) & = \frac 1 N \sum_{x=0}^{N-1} \sum_{y=0}^{N-1}
		f(x,y) cos[\frac \pi N u(x + \frac 1 2)]
		cos[\frac \pi N v(y + \frac 1 2)] \\
	f(x,y) & = \frac 1 N \sum_{u=0}^{N-1} \sum_{v=0}{N-1}
		F(u,v) cos[\frac \pi N u(x + frac 1 2)]
		cos[\frac \pi N v(y + \frac 1 2)]
	\end{align*}$$

-	其他

	$$\begin{align*}
	F(u,v) & = \frac 1 {2N} \sum_{x=0}^{N-1} \sum_{y=0}^{N-1}
		f(x,y) cos[\frac \pi N u(x + \frac 1 2)]
		cos[\frac \pi N v(y + \frac 1 2)] \\
	f(x,y) & = \frac 1 {2N} \sum_{u=0}^{N-1} \sum_{v=0}{N-1}
		F(u,v) cos[\frac \pi N u(x + frac 1 2)]
		cos[\frac \pi N v(y + \frac 1 2)]
	\end{align*}$$



