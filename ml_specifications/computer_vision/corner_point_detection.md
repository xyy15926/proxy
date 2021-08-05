---
title: 角点检测特征提取
categories:
  - ML Specification
  - Computer Vision
tags:
  - Machine Learning
  - Computer Vision
  - Corner Point Detection
  - Moravec
  - Harris
date: 2019-07-13 12:03:12
updated: 2021-07-16 16:10:12
toc: true
mathjax: true
comments: true
description: 角点检测特征提取
---

##	综述

> - *corner point*：角点，邻域各方向上灰度变化值足够高的点，
	是图像边缘曲线上曲率极大值的点

###	分类

-	基于灰度图像的角点检测
	-	基于梯度：计算边缘曲率判断角点
	-	基于模板：考虑像素邻域点的灰度变化，将领域点亮度对比
		足够大的点定义为角点
	-	基于模板、梯度组合

-	基于二值图像的角点检测：将二值图像作为单独的检测目标，
	可使用各种基于灰度图像的角点检测方法

-	基于轮廓曲线的角点检测：通过角点强度、曲线曲率提取角点

###	思想、步骤

-	使用角点检测算子，对图像每个像素计算
	*cornner response function*值

	$$
	E(u, v) = \sum_{(x,y)} w(x,y)[I(x+u, y+v) - I(x,y)]^2
	$$

	> - $w(x,y)$：*window function*，窗口函数
	> - $I(x,y)$：图像梯度
	> - $E(x,y)$：角点响应函数，体现灰度变化剧烈程度，变化
		程度剧烈则窗口中心就是角点

-	阈值化角点响应函数值	
	-	根据实际情况选择阈值$T$
	-	小于阈值$T$者设置为0

-	在窗口范围内对角点响应函数值进行非极大值抑制
	-	窗口内非响应函数值极大像素点置0

-	获取非零点作为角点

##	*Moravec*
#todo

###	步骤

-	取偏移量$(\Delta x, \Delta y)$为
	$(1,0), (1,1), (0,1), (-1,1)$，分别计算每个像素点灰度
	变化

-	对每个像素点(x_i, y_i)$计算角点响应函数
	$R(x) = min \{E\}$

-	设定阈值$T$，小于阈值者置0

-	进行非极大值抑制，选择非0点作为角点检测结果

###	特点

-	二值窗口函数：角点响应函数不够光滑
-	只在4个方向（偏移量）上计算灰度值变化：角点响应函数会在
	多处都有较大响应值
-	对每个点只考虑响应函数值最小值：算法对边缘敏感

##	*Harris*

##	*Good Features to Track*

##	*Feature from Accelerated Segment Test*

*FAST*：加速分割测试获得特征




