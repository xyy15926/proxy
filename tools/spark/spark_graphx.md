---
title: Spark GraphX
tags:
  - 工具
  - Spark
categories:
  - 工具
  - Spark
date: 2019-07-11 00:51:41
updated: 2019-07-11 00:51:41
toc: true
mathjax: true
comments: true
description: Spark GraphX
---

##	GraphX

*Spark GraphX*：图数据的并行处理模块

-	GraphX扩展RDD为*Resilient Distributed Property Graph*，
	边、顶点都有属性的有向图

-	可利用此模块对图数据进行ExploratoryAnalysis、Iterative
	Graph Computation

-	GraphX提供了包括：子图、顶点连接、信息聚合等操作在内的
	基础原语，并且对的Pregel API提供了优化变量的

-	GraphX包括了正在不断增加的一系列图算法、构建方法，用于
	简化图形分析任务

	-	提供了一系列操作
		-	Sub Graph：子图
		-	Join Vertices：顶点连接
		-	Aggregate Message：消息聚集
		-	Pregel API变种

	-	经典图处理算法
		-	PageRank

