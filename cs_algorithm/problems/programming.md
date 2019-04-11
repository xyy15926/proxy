#	规划算法

##	总述

##	线性规划

参见线性规划

###	Simplex Method

单纯型法

####	算法

-	初始化：标准化线性规划问题，建立初始表格

	-	最小化目标函数：目标函数系数取反，求极大

	-	不等式约束：加入松弛变量（代表不等式两端差值）

	-	变量非负：定义为两个非负变量之差

-	最优测试

	-	若目标行系数都为非负，得到最优解，迭代停止
	-	基变量解在右端列中，非基变量解为0

-	确定主元列

	-	从目标行的前n个单元格中选择一个非负单元格，确定
		主元列

	-	选择首个非负：解稳定，若存在最优解总是能取到

	-	选择绝对值最大：目标函数下降快，但有可能陷入死循环，
		无法得到最优解（不满足最优条件）

-	确定主元（分离变量）（行）

	-	对主元列所有正系数，计算右端项和其比值$\Theta$比率

	-	最小$\Theta$比率确定主元（行）（类似的为避免死循环，
		总是选择首个最小者）

-	转轴变换（建立新单纯形表）

	-	主元变1：主元行所有变量除以主元
	-	主元列变0：其余行减去其主元列倍主元行
	-	交换基变量：主元行变量标记为主元列对应变量

####	特点

-	算法时间效率

	-	极点规模随着问题规模指数增长，所以最差效率是指数级

	-	实际应用表明，对m个约束、n个变量的问题，算法迭代次数
		在m到3m之间，每次迭代次数正比于nm

-	迭代改进

###	Two-Phase Simplex Method

两阶段单纯形法：单纯型表中没有单元矩阵，无法方便找到
基本可行解时使用

-	在给定问题的约束等式中加入人工变量，使得新问题具有明显
	可行解
-	利用单纯形法求解最小化新的线性规划问题

###	大M算法

###	Ellipsoid Method

椭球算法


####	特点

-	算法时间效率
	-	可以在多项式时间内对任意线性规划问题求解
	-	实际应用效果较单纯形法差，但是最差效率更好

###	Karmarkar算法

####	特点

-	内点法（迭代改进）

##	Integer Linear Programming Problem

求线性函数的最值，函数包含若干**整数变量**，并且满足线性等式
、不等式的有限约束





