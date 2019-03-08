#	EM算法

###	总述

*expectation maximization algorithm*：含有隐变量的概率模型
参数的极大似然估计法、极大后验概率估计法

####	适合场景

-	模型变量都是*observable variable*、给定数据情况下，可以
	直接使用极大似然估计、贝叶斯估计，但当模型含有
	*latent variable*时不能直接使用

-	模型中含有隐变量时，似然函数将没有解析解，所以EM算法需要
	迭代求解，每次迭代由两步组成
	-	E步：求期望expectation
	-	M步：求极大maximization

####	思想

-	根据已经给出的观测数据，估计模型参数值
-	依据上步估计出的参数值，估计缺失数据值
-	再根据估计储地缺失数据，加上之前已经观测到数据重新对
	参数值进行估计
-	反复迭代直至待估参数收敛

####	EM算法优点

-	EM算法可以用于估计含有隐变量的模型参数
-	非常简单，稳定上升的步骤能非常可靠的找到最优估计值
-	应用广泛，能应用在多个领域中

####	EM算法缺点

-	EM算法计算复杂、受外较慢，不适合高维数据、大规模数据集
-	参数估计结果依赖初值，不够稳定

###	EM算法步骤

> - $\mathcal{Y}$：观测变量数据
> - $\mathcal{Z}$：隐变量数据（未知）
> - $\mathcal{P(Y,Z|\theta)}$：联合分布
> - $\mathcal{P(Z|Y,\theta)}$：条件分布（给定观测数据
	$\mathcal{Y}$$、当前参数估计$\mathcal{\theta}$）
> - $\mathcal{\theta}$：待估参数（参数向量）
> - Q函数：完全数据的对数似然函数$\mathcal{logP(Y,Z|\theta}$
	关于在给定观测$\mathcal{Y}$和当前参数
	$\mathcal{\theta^{(i)}}$下，对未观测数据Z的条件概率分布
	$\mathcal{P(Z|Y,\theta^{(i)})}$
	$$
	Q(\theta, \theta^{(0)}) = E_z\mathcal{
		[logP(Y,Z|\theta)|Y,\theta^{(i)}]}
	$$

-	选择参数初值$\theta^{0}$，开始迭代

	-	算法初值可以任意选择，但EM算法对初值敏感

-	E步：记$\theta^{(i)}$为第$i$迭代时，参数
	$\theta$的估计值，在第$i+1$步迭代的E步时，计算Q函数

	$$\begin{align*}
	Q(\theta, \theta^{(0)}) & = E_z\mathcal{
		[logP(Y,Z|\theta)|Y,\theta^{(i)}]} \\
		& = \sum_Z logP(Y,Z|\theta)P(Z|Y,\theta^{(i)})
	\end{align*}$$

	> - $\theta$：要极大化的参数
	> - $\theta^{(i)}$：参数当前估计值

-	M步：求使得$Q(\theta, \theta^{(i)})$极大化
	$\theta$作为第$i+1$次估计值
	$\theta^{(i+1)}$

	-	每次迭代使得似然函数增大（或达到局部极值）

-	重复E步、M步知道待估参数收敛

	-	收敛条件一般是对较小正数$\epsilon$，满足
		$\|\theta^{(i+1)} - \theta^{(i)}\| < \epsilon$或
		$\|Q(\theta^{(i+1)},\theta^{(i)}) - Q(\theta^{(i)},\theta^{(i)}\| < \epsilon$








