---
title: 矩阵
tags:
  - 代数
  - 线性代数
categories:
  - 代数
  - 线性代数
date: 2019-07-29 21:16:01
updated: 2019-07-29 21:16:01
toc: true
mathjax: true
comments: true
description: 矩阵
---

##	特殊矩阵

-	其中正交矩阵、三角阵、对角阵也被成为因子矩阵

###	Orthogonal Matrix

正交矩阵：和其转置乘积为单位阵的方阵

> - 酉矩阵/幺正矩阵：n个列向量是U空间标准正交基的n阶复方阵，
	是正交矩阵往复数域上的推广

####	几何意义

-	乘正交矩阵：等价于旋转

	![orthogonal_matrix_geo](imgs/orthogonal_matrix_geo.png)

###	Diagonal Matrix

对角阵：仅对角线非0的矩阵

####	几何意义

-	乘对角阵：等价于对坐标轴缩放

	![diagonal_matrix_geo](imgs/diagonal_matrix_geo.png)

###	Triangular Matrix

上/下三角矩阵：左下/右上角全为0的方阵

-	三角阵是高斯消元法的中间产物，方便进行化简、逐层迭代求解
	线性方程组

####	几何意义

-	乘上三角阵：等价于进行右上切变（水平斜拉）

	![upper_triangular_matrix_geo](imgs/upper_triangular_matrix_geo.png)

-	乘下三角阵：等价于进行左下切变（竖直斜拉）

	![lower_triangular_matrix_geo](imgs/lower_triangular_matrix_geo.png)

###	Transposation Matrix

置换矩阵：系数只由0、1组成，每行、列恰好有一个1的方阵

##	矩阵常用公式

###	*Sherman-Morrison*公式

> - 设A是n阶可逆矩阵，$u, v$均为n为向量，若
	$1 + v^T A^{-1} u \neq 0$，则扰动后矩阵$A + u v^T$可逆
	$$
	(A + u v^T)^{-1} = A^{-1} - \frac {A^{-1} u v^T A^{-1}}
		{1 + v^T A^{-1} u}
	$$







