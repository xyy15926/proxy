---
title: Infomation Security
tags:
  - 算法
categories:
  - 算法
date: 2019-05-25 19:55:48
updated: 2019-05-25 19:55:48
toc: true
mathjax: true
comments: true
description: Infomation Security
---

##	Hash/摘要方法

###	文件校验

####	*MD4*

-	附加填充bits：在末尾对消息填充，使得消息$M$长度满足
	$len(M) mod 512 = 448$
	-	填充最高位位为1、其余为0

-	分块：将填充后消息512bits分块为$M_1,M_2,\cdots,M_K$

-	初始化MD4缓存值
	-	MD4使用128bits缓存存储哈希函数中间、最终摘要
	-	将其视为4个32bits寄存器初始化为
		-	`A = 0x67452301`
		-	`B = 0xefcbab89`
		-	`C = 0x98badcfe`
		-	`D = 0x10325476`

-	使用压缩函数迭代计算K个消息分块、上次计算结果
	-	$H_{i+1} = C(H_i, M_i)$
	-	最终$H_K$即为MD4摘要值

####	*MD5*

####	*SHA*

###	数字签名

###	鉴权协议

