---
title: Convolutional
tags:
  - Models
  - Components
  - Machine Learning
  - Convolutional
categories:
  - ML Models
  - Model Components
date: 2019-07-29 21:16:01
updated: 2019-07-29 21:16:01
toc: true
mathjax: true
comments: true
description: Convolutional
---

##	Convolutional

卷积：卷积区域逐点乘积、求和作为卷积中心取值

-	用途：
	-	提取**更高层次**的特征，对图像作局部变换、但保留局部特征
	-	选择和其类似信号、过滤掉其他信号、探测局部是否有相应模式，如
		-	*sobel* 算子获取图像边缘

-	可变卷积核与传统卷积核区别
	-	传统卷积核参数人为确定，用于提取确定的信息
	-	可变卷积核通过训练学习参数，以得到效果更好卷积核

> - 卷积类似向量内积

###	特点

-	局部感知：卷积核所覆盖的像素只是小部分、局部特征
	-	类似于生物视觉中的 *receptive field*

-	多核卷核：卷积核代表、提取某特征，多各卷积核获取不同特征

-	权值共享：给定通道、卷积核，共用滤波器参数
	-	卷积层的参数取决于：卷积核、通道数
	-	参数量远小于全连接神经网络

> - *receptive field*：感受野，视觉皮层中对视野小区域单独反应的神经元
> > -	相邻细胞具有相似和重叠的感受野
> > -	感受野大小、位置在皮层之间系统地变化，形成完整的视觉空间图

###	发展历程

-	1980 年 *neocognitron* 新认知机提出
	-	第一个初始卷积神经网络，是感受野感念在人工神经网络首次应用
	-	将视觉模式分解成许多子模式（特征），然后进入分层递阶式的特征平面处理

##	卷积应用

###	*Guassian Convolutional Kernel*

高斯卷积核：是实现 **尺度变换** 的唯一线性核

$$\begin{align*}
L(x, y, \sigma) & = G(x, y, \sigma) * I(x, y) \\
G(x, y, \sigma) & = \frac 1 {2\pi\sigma^2}
	exp\{\frac {-((x-x_0)^2 + (y-y_0)^2)} {2\sigma^2} \}
\end{align*}$$

> - $G(x,y,\sigma)$：尺度可变高斯函数
> - $I(x,y)$：放缩比例，保证卷积核中各点权重和为 1
> - $(x,y)$：卷积核中各点空间坐标
> - $\sigma$：尺度变化参数，越大图像的越平滑、尺度越粗糙
