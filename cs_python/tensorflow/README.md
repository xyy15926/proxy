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

###	Tensorflow

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

