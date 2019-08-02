---
title: Bagging
tags:
  - 模型
  - 增强模型
categories:
  - 模型
  - 增强模型
date: 2019-07-21 00:46:35
updated: 2019-07-21 00:46:35
toc: true
mathjax: true
comments: true
description: Bagging
---

##	*Bagging*

*bagging*：*bootstrap aggregating*，每个分类器随机从原样本
中做**有放回的随机抽样**，在抽样结果上训练基模型，最后根据
多个基模型的预测结果产生最终结果

-	核心为bootstrap重抽样自举

###	步骤

-	建模阶段：通过boostrap技术获得k个自举样本
	$S_1, S_2,..., S_K$，以其为基础建立k个相同类型模型
	$T_1, T_2,..., T_K$

-	预测阶段：组合K个预测模型
	-	分类问题：K个预测模型“投票”
	-	回归问题：K个预测模型平均值

###	模型性质

-	相较于单个基学习器，Bagging的优势
	-	分类Bagging几乎是最优的贝叶斯分类器
	-	回归Bagging可以通过降低方差（主要）降低均方误差

####	预测误差

总有部分观测未参与建模，预测误差估计偏乐观

-	*OOB*预测误差：*out of bag*，基于袋外观测的预测误差，
	对每个模型，使用没有参与建立模型的样本进行预测，计算预测
	误差

-	OOB观测比率：样本总量n较大时有
	$$
	r = (1 - \frac 1 n)^n \approx \frac 1 e = 0.367
	$$

	-	每次训练样本比率小于10交叉验证的90%




