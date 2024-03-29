---
title: Local Binary Pattern
categories:
  - ML Specification
  - Computer Vision
tags:
  - Machine Learning
  - Computer Vision
  - Local Binary Pattern
  - LBP
date: 2019-07-13 12:03:12
updated: 2021-07-16 16:12:15
toc: true
mathjax: true
comments: true
description: Local Binary Pattern
---

##	综述

局部二值模式：描述图像局部纹理的特征算子

-	具有旋转不变性、灰度不变性
-	通过对窗口中心的、领域点关系进行比较，重新编码形成新特征
	以消除外界场景对图像影响，一定程度上解决了复杂场景下
	（光照变换）特征描述问题
-	分类
	-	经典LBP：3 * 3正方向窗口
	-	圆形LBP：任意圆形领域

##	Classical LBP

###	*Sobel Operator*

###	*Laplace Operator*

###	*Canny Edge Detector*

##	Circular LBP

##	缩略图Hash

-	对图像进行特征提取得到0、1特征向量
-	通过比较图片向量特征间汉明距离即可计算图片之间相似度

###	*Average Hashing*

*aHash*：平均哈希算法

-	将原始图片转换为64维0、1向量，即提取出的特征

####	步骤

-	缩放图片：将图像缩放到8 * 8=64像素
	-	保留结构、去掉细节
	-	去除大小、纵横比差异
-	灰度化：把缩放后图转换为256阶灰度图
-	计算平均值：计算灰度图像素点平均值
-	二值化：遍历64个像素点，大于平均值者记为1、否则为0

###	*Perceptual Hashing*

*pHash*：感知哈希算法

-	利用离散余弦变换降低频率，去除成分较少的高频特征

-	特点
	-	相较于aHash更稳定

####	步骤

-	缩放图片：将图片缩放至32 * 32
-	灰度化：将缩放后图片转换为256阶灰度图
-	计算DCT：把图片分离成频率集合
-	缩小DCT：保留32 * 32左上角8 * 8代表图片最低频率
-	计算平均值：计算缩小DCT均值
-	二值化：遍历64个点，大于平均值者记为1、否则为0

###	*Differential Hashing*

*dHash*：差异哈希算法

-	基于渐变实现

-	特点
	-	相较于dHash非常快
	-	相较于aHash效果好

####	步骤

-	缩放图片：将图片缩放至9 * 8
-	灰度化：将缩放后图片转换为256阶灰度图
-	计算差异值：对每行像素计算和左侧像素点差异，得到8 * 8
-	二值化：遍历64个点，大于0记为1、否则为0

