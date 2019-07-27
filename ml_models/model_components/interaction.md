#	Interaction Layers

##	人工交互作用层

交互作用层：人工设置特征之间交互方式

###	*Flatten Layer*

展平层：直接拼接特征，交互作用交由之后网络训练

$$
f_{Flat}(V_x) = \begin{bmatrix} x_1 v_1 \\ \vdots\\ x_M v_M
	\end{bmatrix}
$$

> - $V_x$：特征向量集合

-	对同特征域特征处理方式
	-	平均
	-	最大

###	二阶交互作用

二阶交互作用层：特征向量之间两两逐元素交互

-	交互方式
	-	逐元素
		-	乘积
		-	求最大值：无
	-	按向量
-	聚合方式
	-	求和
		-	平权
		-	Attention加权
	-	求最大值：无

####	*Bi-Interaction Layer*

*Bi-Interaction Layer*：特征向量两两之间逐元素乘积、求和

$$\begin{align*}
f_{BI}(V) & = \sum_{i=1}^M \sum_{j=i+1}^M v_i \odot v_j \\
& = \frac 1 2 (\|\sum_{i=1}^M v_i\|_2^2 -
	\sum_{i=1}^M \|v_i\|_2^2)
\end{align*}$$

> - $\odot$：逐元素乘积

-	没有引入额外参数，可在线性时间$\in O(kM_x)$内计算
-	可在低层次捕获二阶交互影响，较拼接操作更informative
	-	方便学习更高阶特征交互
	-	模型实际中更容易训练

####	*Attention-based Pooling*

*Attention-based Pooling*：特征向量两两之间逐元素乘积、加权
求和

$$
f_{AP}(V) & = \sum_{i=1}^M \sum_{j=i+1}^M \alpha_{i,j}
	(v_i \odot v_j)
$$

> - $\alpha_{i,j}$：交互作用注意力权重，通过注意力网络训练





