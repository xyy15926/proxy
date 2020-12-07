---
title: Feature Engineering
categories:
  - ML Techs
  - Data Handling
tags:
  - Machine Learning
  - Data Handling
  - Feature Engineering
date: 2019-07-21 00:46:35
updated: 2020-11-03 20:17:42
toc: true
mathjax: true
comments: true
description: Feature Engineering
---

##	综述

特征工程：对原始数据进行一系列工程处理，将其提炼为特征，作为
输入供算法、模型使用

-	本质上：表示、展示数据的过程

-	目的：去除原始数据中的杂质、冗余，设计更高效的特征以刻画
	求解的问题、预测模型之间的关系
	-	把原始数据转换为可以很好描述数据特征
	-	建立在其上的模型性能接近最优

-	方式：**利用数据领域相关知识**、**人为设计输入变量**

-	特征工程重要性：特征越好

	-	模型选择灵活性越高：较好特征在简单模型上也能有较好
		效果，允许选择简单模型
	-	模型构建越简单：较好特征即使在超参不是最优时效果也
		不错，不需要花时间寻找最优参数
	-	模型性能越好
		-	排除噪声特征
		-	避免过拟合
		-	模型训练、预测更快

	> - 数据、特征决定了机器学习的上限，模型、算法只是逼近
		上限

