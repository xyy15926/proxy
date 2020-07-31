---
title: Auto-Encoders
tags:
  - 模型
  - 无监督模型
categories:
  - 模型
  - 无监督模型
date: 2019-07-13 23:26:47
updated: 2019-07-13 12:03:12
toc: true
mathjax: true
comments: true
description: Auto-Encoders
---

自编码机/稀疏编码/堆栈自编码器

-	起源：编码理论可以应用于视觉皮层感受野，大脑主要视觉皮层
	使用稀疏原理创建可以用于重建输入图像的最小基函数子集

-	优点
	-	简单技术：重建输入
	-	可堆栈多层
	-	直觉型，基于神经科学研究

-	缺点
	-	贪婪训练每层
	-	没有全局优化
	-	表现较监督学习差
	-	多层容易失效
	-	输入的重建可能不是学习通用表征的理想*metric*

