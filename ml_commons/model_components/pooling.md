---
title: Pooling Layers
categories:
  - ML Model
  - Model Component
tags:
  - Machine Learning
  - ML Model
  - Model Component
  - Pooling
date: 2019-07-29 21:16:01
updated: 2019-07-29 21:16:01
toc: true
mathjax: true
comments: true
description: Pooling Layers
---

##	池化/下采样

池化：在每个区域中选择只保留一个值

-	用于减小数据处理量同时保留有用的信息
	-	相邻区域特征类似，单个值能表征特征、同时减少数据量

-	保留值得选择有多种
	-	极值
	-	平均值
	-	全局最大

-	直观上
	-	模糊图像，丢掉一些不重要的细节

###	Max Pooling

最大值采样：使用区域中最大值作为代表

###	Average Pooling

平均值采样：使用池中平均值作为代表

