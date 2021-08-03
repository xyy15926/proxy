---
title: Tez
categories:
  - DataBase
  - Hadoop
tags:
  - Hadoop
  - Tez
date: 2019-03-21 17:27:37
updated: 2019-02-17 11:57:08
toc: true
mathjax: true
comments: true
description: Tez
---

##	Tez简介

Tezm目标就是建立执行框架，支持大数据上DAG表达的作业处理

-	YARN将资源管理功能从数据处理模型中独立出来，使得在Hadoop
	执行DAG表达的作业处理成为可能，Tez成为可扩展、高效的执行
	引擎

	-	Tez在YARN和Hive、Pig之间提供一种通用数据处理模型DAG
	-	Hive、Pig、Cascading作业在Tez上执行更快，提供交互式
		查询响应

-	Tez把DAG的每个顶点建模为Input、Processer、Output模块的
	组合

	-	Input、Output决定数据格式、输入、输出
	-	Processor包装了数据转换、处理逻辑
	-	Processor通过Input从其他顶点、管道获取数据输入，通过
		Output向其他顶点、管道传送生成数据
	-	通过把不同的Input、Processor、Output模块组合成顶点，
		建立DAG数据处理工作流，执行特定数据处理逻辑

-	Tez自动把DAG映射到物理资源，将其逻辑表示扩展为物理表示，
	并执行其中的业务逻辑

	-	Tez能为每个节点增加并行性，即使用多个任务执行节点
		的计算任务

	-	Tez能动态地优化DAG执行过程，能够根据执行过程中获得地
		信息，如：数据采样，优化DAG执行计划，优化资源使用、
		提高性能

		-	去除了连续作业之间的写屏障
		-	去除了工作流中多余的Map阶段

###	Tez执行过程

-	初始化例程，提供context/configuration信息给Tez runtime

-	对每个顶点的每个任务（任务数量根据并行度创建）进行初始化

-	执行任务Processor直到所有任务完成，则节点完成

-	Output把从Processor接收到的数据，通过管道传递给下游顶点
	的Input

-	直到整个DAG所有节点任务执行完毕

