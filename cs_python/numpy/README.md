---
title: Numpy约定
tags:
  - Python
  - Numpy
categories:
  - Python
  - Numpy
date: 2019-03-21 17:27:37
updated: 2019-02-17 11:57:07
toc: true
mathjax: true
comments: true
description: Numpy约定
---

##	常用参数说明

-	函数书写说明同Python全局
-	以下常用参数如不特殊注明，按照此解释

###	Ndarray

-	`size=None(1)/int/tuple(int)`

	-	含义：ndarray形状
	-	默认：一般`None`，返回一个值

-	`dtype=None/str/np.int/np.float...`

	-	含义：ndarray中数据类型
	-	默认值：`None`，有内部操作，选择合适、不影响精度类型
	-	其他
		-	可以是字符串形式，也可以是`np.`对象形式
-	`order = "C"/"F"/"K"/"A"`
	-	含义：NDA对象在内存中的存储方式
		-	"C"：`C`存储方式，行优先
		-	"F"：`Fortran`存储方式，列优先
		-	"K"：原为"C"/"F"方式则保持不变，否则按照较接近
			方式
		-	"A"：除非原为"F"方式，否则为"C"方式
	-	默认值："C"/"K"

>	Numpy包中大部分应该是调用底层包，参数形式不好确认



