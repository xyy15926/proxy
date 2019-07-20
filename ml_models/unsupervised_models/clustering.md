#	聚类

##	K-means

最常用的聚类分析

###	发展

-	基于密度
	-	DBSCAN

-	高维数据：data in sparse

-	Graph theorethic clustering

-	Clustering for unstructrued data

-	Large-scale clustering

-	Multi-way clustering

-	clustering for hetergeneous data

###	特点

-	同聚类中对象相似度较高
	不同聚类中对象相似度较低

-	优势在于简洁快速
-	缺点
	-	需要输入参数：k-聚类数目
	-	结果不稳定

-	影响结果因素
	-	原始问题是否可分
	-	分类数目k
	-	初始点选择

-	K值选择
	-	经验选择
	-	特殊方法：Elbow Method，肘部法则，画出距离和K的点图，
		选择剧烈变化的点的K值

###	规则

####	指定k个组

-	数据：$\Omega={X_1, X_2, \dots, X_n}$
-	每个样本点包含p个特征：$X_i = (x_1, x_2, \dots, x_p)$
-	分k个组
	$$
	C_1, C_2, \dots, C_k \\
	C_1 \cup C_2 \cup \dots \cup C_k = \Omega \\
	$$
-	目标：极小化每个样本点到聚类中心距离之和
	$$
	\arg_{C_1, C_2, \dots, C_k} \min \sum_{i=1}^k
		\sum_{x_j in \C_i} d(x_j, C_i)
	$$

	-	若定义距离为平方欧式距离，则根据组间+组内=全，
		极小化目标就是中心点距离极大化

-	优化问题是NP-hard问题，需要采用近似方法

###	Lloyd's Algorithm

####	步骤

-	选择k个初始聚类中心
-	计算每个样本点到k个聚类中心的距离，把样本点分到最近的
	聚类中心对应的类
	Assignmet Step
-	重新计算新类的聚类中心（算术平均），重复以上步骤直到收敛
	（最终聚类中不一定是样本点）
	Update Step

####	算法特点

-	算法时间效率：$\in O(kn^{pi})$

###	应用

####	用户细分

-	Goal: Subdivede a market into distinct subsets of
	customers. Each subset: a market target to be reached
	with a distinct marketing mix

-	Approach: 

##	Fuzzy C-means(FCM)

*FCM*：对K-means的推广

-	*soft cluster*：点可以属于多个类

###	FCM Algorithm

-	初始步骤
-	Assignment Step：计算每个样本点到k个聚类中新的距离，
	得到权重
	$$
	w_k(x_i) = \frac 1 \sum_{i=1}^k 
		(\frac {d(x_i, \mu_k)} {d(x_i, \mu_i} )^{1/(m-2)})
	$$

##	Hierachical Cluster

层次聚类

###	距离

$dist(x,y)$：不一定是空间距离，应该认为是两个对象x、y之间的
相似程度

-	欧几里得距离
-	切比雪夫距离
-	闵科夫斯基距离
-	曼哈顿距离
-	余弦相似度

###	方法

####	AGENS

#####	Procedure

-	grouped in a bottom-up fashion
-	initally all data is in its own cluster

#####	

-	算法复杂度：$n^2logn$

-	组连接：组与组之间距离
	-	single linkage
	-	average linkage
	-	complete linkage

####	DIANA

Divisive Analysis：想法简单，具体实操有难度

#####	Procedure

-	grouped in a top-down manner

-	初始：所有数据归为一组$C_1=(p_1, p_2, dots, p_n)$
-	2：计算所有点之间的距离矩阵，选择到其他点平均距离最大的
	点，记为$q$，取该点作为新组起始点
-	3：$\forall p, p \notin C_1$，计算$d_arg(p, C_1) - d_arg(p, C_2)$，
	若小于零则属于$C_1$，否则属于$C_2$

##	*Density-based Scan*

*DBSCAN*：

##	*Balanced Itertive Reducing and Clustering Using Hierarchies*

*BIRCH*：利用层次方法的平衡迭代规约和聚类，利用层次方法聚类
、规约数据

-	特点
	-	利用CF树结构快速聚类
	-	只需要单遍扫描数据

