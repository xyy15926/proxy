---
title: 无向图
tags:
  - 算法
  - 数据结构
categories:
  - 算法
  - 数据结构
date: 2019-03-23 23:43:02
updated: 2019-03-23 23:43:02
toc: true
mathjax: true
comments: true
description: 无向图
---

##	点集

###	点覆盖集

-	*Vertex Covering Set*：点覆盖（集），顶点子集$S \subseteq V$
	，满足每条边至少有一个顶点在集合中

-	*Minimum Vertex Covering Set*：最小顶点覆盖集，最少顶点的
	点覆盖集

###	点支配集

-	*Vertex Dominating Set*：点支配集，顶点子集$D \subseteq V$
	，满足$\forall u \in V-D, \exists v \in D, (u, v) \in E$

	-	即V中顶点要么是D中元素，要么同D中一个顶点相邻

-	*Minimum Vertext Dominating Set*：最小支配集，顶点数目
	最小支配集

-	极小支配集：真子集都不是支配集的支配集

###	点独立集

-	*Vertext Independent Set*：（点）独立集，顶点子集
	$I \subseteq V$，满足I中任意两顶点不相邻

-	*Maximum Vertext Independent Set*：最大独立点集，顶点数
	最多的独立点集

-	极大点独立集：超集都不是独立集的独立集

###	性质

####	Thoerem 1

若无向图$G(V, E)$中无孤立顶点，则G的极大点独立集都是G的极小
支配集（反之不成立）

####	Thoerem 2

一个独立集是极大独立集，当前且仅当其为支配集

####	Thoerem 3

若无向图$G=(V, E)$中无孤立点，顶点集$C \subseteq V$为G点覆盖
，当且仅当$V - C$是G的点独立集

##	边集

###	边覆盖

-	*Edge Covering Set*：边覆盖（集），边子集$W \subseteq E$，
	满足$\forall v \in V, \exists e \in W$，使得v是e端点

	-	即G中所有点都是便覆盖W的邻接顶点

-	*Minimum Edge Covering Set*：边数最少的边覆盖集

-	极小边覆盖集：任意真子集都不是边覆盖集的边覆盖

###	边独立集

-	*Matching/Edge Indepdent Set*：匹配（边独立集），边子集
	$I \subseteq E$，满足I中所有边没有公共顶点

-	*Maximum (Cardinality) Matching*：最大（基数）匹配，包含
	最多边的匹配

-	极大匹配：任意超集都不是匹配的匹配

-	*Perfect Matching*：完美匹配，匹配所有点的最大匹配

-	*Mate*：对偶，匹配中相互连接的一对顶点

###	性质

####	Thoerem 1

M为G一个最大匹配，对G中每个M未覆盖点v，选取一条与v关联边组成
集合N，则边集$W = M \cup N$为G中最小边覆盖

####	Thoerem 2

若W为G最小边覆盖，其中每存在相邻边就移去其中一条，设移去边集
为N，则边集$M = W - N$为G中一个最大匹配

####	Thoerem 3

最大匹配、最小边覆盖满足：$|M| + |W|= |V|$

