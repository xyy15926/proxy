#	距离函数

$dist(x,y)$：不一定是空间距离，应该认为是两个对象x、y之间的
**相似程度**

-	距离和相似度是互补的
-	可以根据处理问题的情况，自定义距离

##	单点距离

###	*Minkowski Distance*

闵科夫斯基距离：向量空间$\mathcal{L_p}$范数

$$
d_{12} = \sqrt [1/p] {\sum_{k=1}^n |x_{1,k} - x_{2,k}|^p}
$$

-	表示一组距离族

	-	$p=1$：*Manhattan Distance*，曼哈顿距离
	-	$p=2$：*Euclidean Distance*，欧式距离
	-	$p \rightarrow \infty$：*Chebychev Distance*，
		切比雪夫距离

-	闵氏距离缺陷

	-	将各个分量量纲视作相同
	-	未考虑各个分量的分布

###	*Mahalanobis Distance*

马氏距离：表示数据的协方差距离

$$
d_{12} = \sqrt {({x_1-\mu}^T) \Sigma^{-1} (x_2-\mu)}
$$

> - $\Sigma$：总体协方差矩阵

-	优点

	-	马氏距离和原始数据量纲无关
	-	考虑变量相关性

-	缺点

	-	需要知道总体协方差矩阵，使用样本估计效果不好

###	*LW Distance*

兰氏距离：*Lance and Williams Distance*，堪培拉距离

$$
d_{12} = \sum^{n}_{k=1} \frac {|x_{1,k} - x_{2,k}|}
	{|x_{1,k} + x_{2,k}|}
$$

-	对接近0的值非常敏感
-	对量纲不敏感
-	未考虑变量直接相关性，认为变量之间相互独立

###	*Hamming Distance*

汉明距离：差别

$$
diff = \frac 1 p \sum_{i=1}^p  (v^{(1)}_i - v^{(2)}_i)^k 
$$

> - $v_i \in \{0, 1\}$：虚拟变量
> - $p$：虚拟变量数量

-	可以衡量定性变量之间的距离

####	Embedding

-	找到所有点、所有维度坐标值中最大值$C$
-	对每个点$P=(x_1, x_2, \cdots, x_d)$
	-	将每维$x_i$转换为长度为$C$的0、1序列
	-	其中前$x_i$个值为1，之后为0
-	将$d$个长度为$C$的序列连接，形成长度为$d * C$的序列

> - 以上明距离空间嵌入对曼哈顿距离是保距的

###	*Jaccard*系数

*Jaccard*系数：度量两个集合的相似度，值越大相似度越高

$$
sim = \frac {\|S_1 \hat S_2\|} {\|S_1 \cup S_2\|}
$$

> - $S_1, S_2$：待度量相似度的两个集合

###	*Adamic-Aria*

$$
AA = \sum_{i \in \Dao(x) \cap \Dao (y)} log \frac 1 Degress(i)
$$
#todo

###	*Consine Similarity*

余弦相似度

$$
simimarity = cos(\theta) = \frac {x_1 x_2} {\|x_1\|\|x_2\|}
$$

###	*Levenshtein/Edit Distance*

（字符串）编辑距离：两个字符串转换需要进行插入、删除、替换
操作的次数

$$
lev_{A,B}(i, j) = \left \{ \begin{array}{l}
	i, & j = 0 \\
	j, & i = 0 \\
	min \left \{ \begin{array}{l}
		lev_{A,B}(i,j-1) + 1 \\
		lev_{A,B}(i-1,j) + 1 \\
		lev_{A,B}(i-1, j-1) + 1
	\end{array} \right. & A[i] != B[j] \\
	min \left \{ \begin{array}{l}
		lev_{A,B}(i,j-1) + 1 \\
		lev_{A,B}(i-1,j) + 1 \\
		lev_{A,B}(i-1, j-1)
	\end{array} \right. & A[i] = B[j] \\
\end{array} \right.
$$

##	组间距离

###	*Single Linkage*

###	*Average Linkage*

###	*Complete Linkage*

##	符号距离

> - $T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_n,y_n)\}$：样本点集
> - $wx + b = 0$：超平面

###	*Functional Margin*

函数间隔

$$
\hat{\gamma_i} = y_i(wx_i + b)
$$


-	函数间隔可以表示分类的正确性、确信度

	-	正值表示正确
	-	间隔越大确信度越高

-	点集与超平面的函数间隔取点间隔最小值
	$\hat{T} = \min_{i=1,2,\cdots,n} \hat{\gamma_i}$

-	超平面参数$w, b$成比例改变时，平面未变化，但是函数间隔
	成比例变化

###	*Geometric Margin*

几何间隔

$$\begin{align*}
\gamma_i & = \frac {y_i} {\|w\|} (wx_i + b) \\
	& = \frac {\hat \gamma_i} {\|w\|}
\end{align*}$$

-	几何间隔一般是样本点到超平面的*signed distance*

	-	点正确分类时，几何间隔就是点到直线的距离

-	几何间隔相当于使用$\|w\|$对函数间隔作规范化

	-	$\|w\|=1$时，两者相等
	-	几何间隔对确定超平面、样本点是确定的，不会因为超平面
		表示形式改变而改变

-	点集与超平面的几何间隔取点间隔最小值
	$\hat{T} = \min_{i=1,2,\cdots,n} \hat{\gamma_i}$

