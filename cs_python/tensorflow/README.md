---
title: Tensorflow约定
tags:
  - Python
  - Tensorflow
categories:
  - Python
  - Tensorflow
date: 2019-03-21 17:27:37
updated: 2019-02-17 11:57:07
toc: true
mathjax: true
comments: true
description: Tensorflow约定
---

##	常用参数说明

-	函数书写声明同Python全局

-	以下常用参数如不特殊注明，按照此解释

###	Session

-	`target = ""/str`
	-	含义：执行引擎

-	`graph = None/tf.Graph`
	-	含义：Session中加载的图
	-	默认：缺省为当前默认图

-	`config = None/tf.ConfigProto`
	-	含义：包含Session配置的*Protocal Buffer*
	-	默认：`None`，默认配置

-	`fetches = tf.OPs/[tf.OPs]`
	-	含义：需要获得/计算的OPs值列表
	-	默认：无

-	`feed_dict = None/dict`
	-	含义：替换/赋值Graph中feedable OPs的tensor字典
	-	默认：无
	-	说明
		-	键为图中节点名称、值为向其赋的值
		-	可向所有可赋值OPs传递值
		-	常配合`tf.placeholder`（强制要求）

###	Operators

-	`name = None/str`
	-	含义：Operations名
	-	默认：有的为None，有的为Operation类型，但效果一样，
		没有传参时使用Operation类型（加顺序后缀）

-	`axis = None/0/int`
	-	含义：指定张量轴
	-	默认
		-	`None`：大部分，表示在整个张量上运算
		-	`0`：有些运算难以推广到整个张量，表示在首轴（维）

-	`keepdims=False/True`
	-	含义：是否保持维度数目
	-	默认：`False`不保持

##	TensorFlow基本概念

-	TensorFlow特点
	-	将计算的定义、执行分开

###	流程

####	组合计算图

-	为输入、标签创建placeholder
-	创建weigth、bias
-	指定模型
-	指定损失函数
-	创建Opitmizer

####	在会话中执行图中操作

-	初始化Variable
-	运行优化器
-	使用`FileWriter`记录log
-	查看TensorBoard

