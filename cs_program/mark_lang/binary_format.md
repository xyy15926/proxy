---
title: 二进制文件格式
tags:
  - 程序
categories:
  - 程序
date: 2019-05-10 01:01:26
updated: 2019-05-10 01:01:26
toc: true
mathjax: true
comments: true
description: 二进制文件格式
---

##	*IDX*

*IDX*：MNIST数据集独创的数据格式

-	用于存储多维数组

-	后可以跟数字表示存储数组的维度
	-	*idx1*：存储1维数组
	-	*idx3*：存储3维数组

###	格式

-	2bytes：格式版本号
	-	一直是`0x0000`

-	1bytes：数组中每个元素的数据类型
	-	`0x08`：`unsigned byte`
	-	`0x09`：`signed byte`
	-	`0x0B`：`short`（2bytes）
	-	`0x0C`：`int`（4bytes）
	-	`0x0D`：`float`（4bytes）
	-	`0x0E`：`double`（8bytes）

-	1bytes：数组维度`d`

-	`d` * 4bytes(int)：数组各维度长度

-	数据部分
	-	数据类型已知、长度已知
	-	若元素格式符合文件头要求，表明解析正确，否则文件损坏


