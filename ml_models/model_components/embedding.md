#	Embedding

##	Embedding

嵌入层：将高维空间中离散变量映射为低维稠密embedding向量表示

-	embedding向量更能体现样本之间关联
	-	內积（內积）体现样本之间接近程度
	-	可通过可视化方法体现样本差异

-	embedding向量更适合某些模型训练
	-	模型不适合高维稀疏向量
	-	embedding向量矩阵可以联合模型整体训练，相当于提取
		特征

###	Embedding表示

-	特征不分组表示

	$$\begin{align*}
	\varepsilon_x & =  E x \\
	& = [x_1v_1, x_2v_2, \cdots, x_Mv_M] \\
	& = [x_{M_1} v_{M_1}, \cdots, x_{M_m} v_{M_m}]
	\end{align*}$$

	> - $E$：embedding向量矩阵
	> - $M$：特征数量
	> - $v_i$：$k$维embedding向量
	> - $x_i$：特征取值，对0/1特征仍等价于查表，只需考虑非0特征
	> > -	$x_{M_i}$：第$j$个非0特征，编号为$M_i$
	> > -	$m$：非零特征数量
	> - $\varepsilon_x$：特征向量集合

-	特征分组表示

	$$\begin{align*}
	\varepsilon_x & = [V_1 g_1, V_2 g_2, \cdots, V_G g_G]
	\end{align*}$$

	> - $G$：特征组数量
	> - $V_i$：第$i$特征组特征向量矩阵
	> - $g_i$：第$i$特征组特征取值向量


