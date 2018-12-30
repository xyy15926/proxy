#	Vector Autoregressive Model

##	VAR模型

*VAR*：向量自回归模型

-	模型特点
	-	不以经济理论为基础
	-	结构简介明了
	-	预测精度高
-	模型方程特点
	-	采用多方程联立的形式
	-	不进行参数显著性检验，但是也允许研究人员对参数施加
		特殊约束
	-	需要估计$m(mp+1)$个参数的，对样本数量要求高
	-	模型的每个方程中，**内生变量**对模型的全部内生变量
		滞后项进行回归，估计全部内生变量的动态关系
-	用途
	-	脉冲响应分析
	-	方差分解

###	两变量VAR(1)

-	方程组形式
	$$ \begin{cases}
	& y_{1,t} = c_1 + \pi_{11.1}y_{1,t-1} + \pi_{12.1}y_{2,t-1}
		+ u_{1t} \\
	& y_{2,t} = c_2 + \pi_{21.1}y_{1,t-1} + \pi_{22.1}y_{2,t-1}
		+ u_{2t} \\
	\end{cases}$$

-	矩阵形式
	$$
	\begin{bmatrix} y_{1t} \\ y_{2t} \end{bmatrix} =
	\begin{bmatrix} c_1 \\ c_2 \end{bmatrix} +
	\begin{bmatrix}
		\pi_{11.1} & \pi_{12.1} \\
		\pi_{21.1} & \pi_{22.1} \\
	\end{bmatrix}
	\begin{bmatrix} y_{1,t-1} \\ y_{2,t-1} \end{bmatrix} +
	\begin{bmatrix} u_{1t} \\ u_{2t} \end{bmatrix}
	$$

-	其中
	-	$u_{1t}, u_{2t} \overset {i.i.d.} {~} (0, \theta^2)$
	-	$Cov(u_1, u_2) = 0$

###	N变量的VAR(k)

$$
Y_t = c + \Pi_1 Y_{t-1} + \Pi_2 Y_{t-2} + \cdots + 
	\Pi_k Y_{t-k} + u_t
$$

其中

> - $Y_t = (y_{1,t}, y_{2,t}, \cdots, y_{N,t})^T$
> - $c = (c_1, c_2, \cdots, c_N)^T$
> - $\Pi_j = \begin{bmatrix}
		\pi_{11,j} & \pi_{12,j} & \cdots & \pi_{1N,j} \\
		\pi_{21,j} & \pi_{22,j} & \cdots & \pi_{2N,j} \\
		\vdots & \vdots & \ddots & \vdots \\
		\pi_{N1,j} & \pi_{N2,j} & \cdots & \pi_{NN,j} \\
	\end{bmatrix}$
> - $u_t = (u_{1t}, u_{2t}, \cdots, u_{Nt})^T \overset {i.i.d.} {~} (0, \Omega)$

###	Structured VAR

*SVAR*：结构VAR模型，在VAR模型基础中加入了内生变量当期值

-	即解释变量中含有当期变量

####	两变量SVAR(1)

$$
\begin{cases}
& y_{1t} = c_1 + \pi_{11}y_{2t} + \pi_{12}y_{1,t-1} +
	\pi_{13}y_{1,t-2} + u_1 \\
& y_{2t} = c_2 + \pi_{21}y_{1t} + \pi_{22}y_{1,t-1} +
	\pi_{23}y_{1,t-2} + u_2 \\
\end{cases}
$$

###	含外生变量VAR(1)

$$\begin{align*}
AY_t & = D + BY_{t-1} + FZ_t + v_t \\
Y_t & = A^{-1}D + A^{-1}BY_{t-1} + A^{-1}FZ_t + A^{-1}v_t \\
	& = c + \Pi_1 Y_{t-1} + HZ_t + u_t
\end{align*}
$$


> - $Y_t, Z_t, v_t$：内生变量向量、外生变量向量、误差项向量
> - $A, D, B, F$：模型结构参数
> - $c=A^{-1}D, \Pi_1=A^{-1}B, H=A^{-1}F, u_t=A^{-1}v_t$

##	VAR模型稳定性

模型稳定性：把脉冲施加在VAR模型中某个方程的innovation过程上

> - 随着时间推移，冲击会逐渐消失，则模型稳定
> - 冲击不消失的则模型不稳定

###	一阶VAR模型分析

$$\begin{align*}
Y_t & = c + \Pi_1Y_{t-1} + u_t \\
	& = (I + \Pi_1 + \Pi_2^2 + \cdots + \Pi_1^{t-1})c +
		\Pi_1^tY_0 + \sum_{i=0}^{t-1} \Pi_1^i u_{t-1}
\end{align*}$$

> - $\mu = (I + \Pi_1 + \Pi_2^2 + \cdots + \Pi_1^{t-1})c$：
	漂移向量
> - $Y_0$：初始向量
> - $u_0$：新息向量

$t \leftarrow \infty$时有
$I + \Pi_1 + \Pi_2^2 + \cdots + \Pi_1^{t-1} = (I-\Pi_1)^{-1}$

##	VAR建模

-	进行单变量平稳性检验

-	拟合VAR(p)模型
	-	确定模型阶数：理论上初步模型阶数可以任意确定，然后
		根据AIC、BIC、对数似然函数值选择相对最优阶数

-	若所有变量平稳
	-	Granger因果检验

-	若有变量非平稳
	-	检验模型平稳性
	-	Granger因果检验
	-	协整检验
	-	构建VEC模型

-	脉冲响应分析

-	方差分析

-	模型预测








