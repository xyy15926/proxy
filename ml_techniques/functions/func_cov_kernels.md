---
title: 图像卷积核
tags:
  - 机器学习
categories:
  - 机器学习
date: 2019-07-13 23:45:19
updated: 2019-07-13 12:03:11
toc: true
mathjax: true
comments: true
description: 图像卷积核
---

##	*Guassian Convolutional Kernel*

高斯卷积核：是实现尺度变换的唯一线性核

$$\begin{align*}
L(x, y, \sigma) & = G(x, y, \sigma) * I(x, y) \\
G(x, y, \sigma) & = \frac 1 {2\pi\sigma^2}
	exp\{\frac {-((x-x_0)^2 + (y-y_0)^2)} {2\sigma^2} \}
\end{align*}$$

> - $G(x,y,\sigma)$：尺度可变高斯函数
> - $(x,y)$：空间坐标，尺度坐标
> > -	大尺度：图像概貌特征
> > -	小尺度：图像细节特征
> - $\sigma$：图像的平滑程度
> > -	大$\sigma$：粗糙尺度，低分辨率
> > -	小$\sigma$：精细尺度，高分辨率

