#	GARCH波动性建模

##	白噪声检验

进行ARIMA模型拟合时，通过检验查询序列是否通过LB检验（白噪声
检验）判断模型显著性，但是实际上

-	LB检验只检验了白噪声序列的纯随机性
-	对于方差齐性没有进行检验
-	至于零均值，可以在建模时通过拟合常数项解决

##	方差齐性变换

如果已知异方差函数的具体形式，找到其转换函数，进行方差齐性
变换

$$
若异方差形式为：\sigma_t^2 = h(\mu_t)，寻找转换函数g(x) \\
\Rightarrow g(x) \approx g(\mu_t) + (x_t - \mu_t)g'(\mu_t) \\
\Rightarrow Var[g(x_t) \approx [g'(\mu_t)^2]Var(x_t) \\
\Rightarrow [g'(mu_t)]^2 h(\mu_t) = \sigma^2 \\
\Rightarorw Var[g(x_t)] = \sigma^2 \\
$$

##	拟合条件异方差模型
