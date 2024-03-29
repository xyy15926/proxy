---
title: 数组和广义表
categories:
  - Algorithm
  - Data Structure
tags:
  - Algorithm
  - Data Structure
  - Linear
  - Array
  - List
date: 2019-03-21 17:27:37
updated: 2021-08-02 18:09:32
toc: true
mathjax: true
comments: true
description: 数组和广义表
---

##	综述

-	数组、广义表可以看作是线性表的扩展：线性表中数据元素本身
	也是抽象数据结构

##	*Array*

对各维界分别为$b_i$的n维数组

-	数组中含有$\prod_{i=1}^n b_i$个数据元素，每个元素都受n个
	关系约束
	-	每个关系中，元素都有直接后继
	-	就单个关系而言，这个n个关系仍然是线性关系

-	n维数组可看作是线性表的推广，n=1时，数组退化为定长线性表
	-	和线性表一样，所有数据元素必须为同一数据类型

-	数组一旦被定义，其维数、维界就不再改变
	-	除了初始化、销毁之外，数组只有存储、修改元素值的操作
	-	采用顺序存储结构表示数组是自然的

> - 几乎所有程序设计语言都把数组类型设定为固有类型

###	顺序存储表示

-	存储单元是一维结构，数组是多维结构，所有使用连续存储单元
	存储数组的数据元素需要约定次序

	-	BASIC、PL/1、COBOL、PASCAL、C语言中，以行序作主序
	-	FORTRAN中，以列序作主序

-	数组元素存储地址

	$$\begin{align*}
	LOC(j_1, j_2, \cdots, j_n) & = LOC(0, 0, \cdots, 0)
		+ (\sum_{i=1}^{n-1} j_i (\prod_{k=i+1}^n b_k + j_n))L
	& = LOC(0, 0, \cdots, 0) + \sum_{i=1}^n c_i_i
	\end{align}$$

	> - $c_n=L, c_{i-1} = b_ic_i$

```c
typedef struct{
	Elemtype * base;
	int dim;
		// 数组维数
	int * bounds;
		// 数组各维界（各维度长度）
	int * constants;
		// 数组各维度单位含有的数组元素数量，由`bounds`累乘
}
```

###	矩阵压缩

压缩存储：为多个值相同元只分配一个存储空间，对0元不分配空间

####	特殊矩阵

特殊矩阵：值相同元素、0元素在矩阵的分布有一定规律，将其压缩
至一维数组中，并找到每个非零元在一维数组中的对应关系

-	对称矩阵：为每对对称元分配一个存储空间
	-	一般以行序为主序存储其下三角（包括对角线）中的元
-	上/下三角矩阵：类似对称矩阵只存储上/下三角中的元，附加
	存储下/三角常数
-	对角矩阵：同样可以按照行、列、对角线优先压缩存储

####	稀疏矩阵

稀疏矩阵：稀疏因子$\sigma = \frac t {mn} \leq 0.05$的矩阵

-	使用三元组（非零元值、其所属的行列位置）表示非零元

#####	三元组顺序表

三元组顺序表/有序双下标法：以顺序结构表示三元组表

```c
typedef struct{
	int i, j;
	ElemType e;
}Triple;
typedef struct{
	Triple data[MAXSIZE+1];
	int mu, nu, tu;
		// 行、列、非零元数
}TSMatrix;
```

> - `data`域中非零元的三元组以行序为主序顺序排列，有利于进行
	依行顺序处理的矩阵运算

#####	行逻辑链接的顺序表

行逻辑链接的顺序表：带行链接信息的三元组表

```c
typedef struct{
	Triple data[MAXSIZE+1];
	int rpos[MAXRC+1];
		// 各行首个非零元的位置表
	int mu, nu, tu;
}RLSMatrix;
```

-	为了便于随机存取任意行非零元，需要知道每行首个去非零元
	在三元组表中的位置，即`rpos`

#####	十字链表

十字链表：采用链式存储结构表示三元组的线性表

```c
typedef struct OLNode{
	int i,j;
		// 该非零元行、列下标
	ElemType e;
	struct OLNode, *right, *down;
		// 该非零元所在行表、列表的后继链域
}OLNode, *OLink;
typedef struct{
	OLink *rhead, *chead;
		// 行、列链表表头指针向量基址
	int mu, nu, tu;
}CrossList;
```

> - 同一行非零元通过`right`域链接成一个线性链表
> - 同一列非零元通过`down`域链接成一个线性链表

-	适合矩阵非零元个数、位置在操作过程中变化较大的情况

##	*Lists*

广义表/列表：线性表的推广

$$
LS = (\alpha_1, \alpha_2, \cdots, \alpha_n)
$$

> - $\alpha_i$：可以时单个元素，也可以是广义表，分别称为LS
	的原子、子表
> - *head*：表头，LS非空时的首个元素$\alpha$
> - *tail*：表尾，LS除表头外的其余元素组成的表，必然是列表

-	列表是一个多层次结构：列表的元素可以是子表，子表元素
	也可以是子表
-	列表可以为其他列表所共享
-	列表可以是一个递归的表：即列表自身作为其本身子表

> - 广义表长度：广义表中元素个数
> - 广义表深度：广义表中最大括弧重数

###	链式存储结构

广义表中数据元素可以具有不同结构，难以用顺序存储结构表示，
通常采用链式存储结构

####	头尾链表

-	数据元素可能是原子、列表，因此需要两种结构的结点
	-	表节点：表示列表，包括：标志域、指示表头的指针域、
		指示表尾的指针域
	-	原子节点：表示原子，包括：标志域，值域

```c
typedef enum{ATOM, LIST} ElemTag;

typedef struct GLNode{
	ElemTag tag;
		// 标志域，公用
	union{
		// 原子节点、表节点联合部分
		AtomType atom;
			// 值域，原子节点
		struct{
			struct GLNode *hp, *tp;
				// 两个指针域，表节点
		}ptr;
	};
}*GList;
```

-	除空表的表头指针为空外，任何非空列表的表头指针均指向
	表节点，且该表节点
	-	`hp`指针域指向列表表头
	-	`tp`指针域指向列表表尾，除非表尾为空，否则指向表节点
-	容易分清列表中原子、子表所属层次
-	最高层的表节点个数即为列表长度

####	扩展线性链表

```cpp
typedef enum {ATOM, LIST} ElemTag;

typedef struct GLNode{
	ElemTag tag;
	union{
		AtomType atom;
		struct GLNode *hp;
		};
	struct GLNode *tp;
}*GList;
```

