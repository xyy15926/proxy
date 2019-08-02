---
title: Keras约定
tags:
  - Python
  - Keras
categories:
  - Python
  - Keras
date: 2019-03-21 17:27:37
updated: 2019-02-17 11:57:07
toc: true
mathjax: true
comments: true
description: Keras约定
---

##	常用参数说明

-	函数书写声明同Python全局说明
-	以下常用参数如不特殊注明，按照此解释

###	Common

-	`seed=None/int`
	-	含义：随机数种子

-	`padding="valid"/"same"/"causal"`
	-	含义：补0策略
		-	"valid"：只进行有效有效卷积，忽略边缘数据，输入
			数据比输出数据shape减小
		-	"same"：保留边界处卷积结果，输入数据和数据shape
			相同
		-	"causal"：产生膨胀（因果卷积），即`output[t]`
			不依赖`input[t+1:]`，对不能违反时间顺序的时序
			信号建模时有用
	-	默认：*valid*

###	Layers

-	`input_shape=None/(int,...)`
	-	含义：输入数据shape
		-	Layers只有首层需要传递该参数，之后层可自行推断
		-	传递tuple中`None`表示改维度边长
	-	默认：`None`，由Layers自行推断

-	`data_format=None/"channels_last"/"channels_first"`
	-	含义：通道轴位置
		-	类似于`dim_ordering`，但是是Layer参数
	-	默认
		-	大部分：`None`由配置文件（默认"channels_last"）
		、环境变量决定
		-	Conv1DXX："channels_last"
		-	其实也不一定，最好每次手动指定

-	`dim_ordering=None/"th"/"tf"`
	-	含义：中指定channals轴位置(`th`batch后首、`tf`尾）
	-	默认：`None`以Keras配置为准
	-	注意：Deprecated，Keras1.x中使用


###	Conv Layers

-	`filters(int)`
	-	含义：输出维度
		-	对于卷积层，就是卷积核数目，因为卷积共享卷积核
		-	对于局部连接层，是卷积核**组数**，不共享卷积核
			，实际上对每组有很多不同权重

-	`kernel_size(int/(int)/[int])`
	-	含义：卷积核形状，单值则各方向等长

-	`strides(int/(int)/[int])`
	-	含义：卷积步长，单值则各方向相同
	-	默认：`1`移动一个步长

-	`dilation_rate(int/(int)/[int])`
	-	含义：膨胀比例
		-	即核元素之间距离
		-	`dilation_rate`、`strides`最多只能有一者为1，
			即核膨胀、移动扩张最多只能出现一种
	-	默认：`1`不膨胀，核中个元素相距1

-	`use_bias=True/False`
	-	含义：是否使用偏置项
	-	默认：`True`使用偏置项

-	`activation=str/func`
	-	含义：该层激活函数
		-	`str`：预定义激活函数字符串
		-	`func`：自定义element-wise激活函数
	-	默认：`None`不做处理（即线性激活函数）

-	`kernel_initializer=str/func`
	-	含义：权值初始化方法
		-	`str`：预定义初始化方法名字符串
			（参考Keras Initializer）
		-	`func`：初始化权重的初始化器
	-	默认：`glorot_uniform`初始化为平均值

-	`bias_initializer=str/func`
	-	含义：偏置初始化方法
		-	`str`：预定义初始化方法名字符串
		-	`func`：初始化权重的初始化器
	-	默认：`zeros`初始化为全0

-	`kernel_regularizer=None/obj`
	-	含义：施加在权重上的正则项
		（参考Keras Regularizer对象）
	-	默认：`None`不使用正则化项

-	`bias_regularizer=None/obj`
	-	含义：施加在偏置上的正则项
		（参考Keras Regularizer对象）
	-	默认：`None`不使用正则化项

-	`activity_regularizer=None/obj`
	-	含义：施加在输出上的正则项
		（参考Keras Regularizer对象）
	-	默认：`None`不使用正则化项

-	`kernel_constraint=None/obj`
	-	含义：施加在权重上的约束项
		（参考Keras Constraints）
	-	默认：`None`不使用约束项

-	`bias_constraint=None`
	-	含义：施加在偏置上的约束项
		（参考Keras Constraints）
	-	默认：`None`不使用约束项

