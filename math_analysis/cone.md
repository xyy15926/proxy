#	Cone

##	概念

###	*Cone*

锥

$$\mathcal{
C \subset V, \forall x \in C, ax \in C; a > 0
}$$

> - $\mathcal{V}$：向量空间

-	显然，锥总是无界的

###	*Convex Cone*

凸锥

$$\mathcal{
\forall x,y \in C, ax + by \in C; a,b > 0
}$$

> - $\mathcal{C}$：锥

-	凸锥比凸集范围更大，凸锥必然是凸集
-	非凸锥：凸锥的补集

	-	$\mathcal{y=\|x\|}$：两条射线

###	*Norm Cone*

n维标准锥

$$\mathcal{
C = \{(x,t)| \|x\|_2 \leq t, x \in R^{n-1}, t \in R\}
}$$

###	*Second Order Cone*

二阶锥

$$\mathcal{
C = {(x,t)\|Ax+b\|_2 \leq c^Tx + d
}$$

-	二阶锥相对于对标准锥做了仿射变换（平移变换）

