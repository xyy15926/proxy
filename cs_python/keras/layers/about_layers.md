---
title: Layers 总述
categories:
  - Python
  - Keras
tags:
  - Python
  - Keras
  - Machine Learning
  - Layer
date: 2019-02-20 23:58:15
updated: 2021-08-04 11:42:51
toc: true
mathjax: true
comments: true
description: Layers总述
---

###	Layer方法

所有的Keras层对象都有如下方法：

-	`layer.get_weights()`：返回层的权重NDA

-	`layer.set_weights(weights)`：从NDA中将权重加载到该层中
	，要求NDA的形状与`layer.get_weights()`的形状相同

-	`layer.get_config()`：返回当前层配置信息的字典，层也可以
	借由配置信息重构

-	`layer.from_config(config)`：根据`config`配置信息重构层

	```python
	layer = Dense(32)
	config = layer.get_config()
	reconstructed_layer = Dense.from_config(config)
	```

	```python
	from keras import layers

	config = layer.get_config()
	layer = layers.deserialize({'class_name': layer.__class__.__name__,
								'config': config})
	```

####	非共享层

如果层仅有一个计算节点（即该层不是共享层），则可以通过下列
方法获得

-	输入张量：`layer.input`
-	输出张量：`layer.output`
-	输入数据的形状：`layer.input_shape`
-	输出数据的形状：`layer.output_shape`

####	共享层

如果该层有多个计算节点（参考层计算节点和共享层）

-	输入张量：`layer.get_input_at(node_index)`
-	输出张量：`layer.get_output_at(node_index)`
-	输入数据形状：`layer.get_input_shape_at(node_index)`
-	输出数据形状：`layer.get_output_shape_at(node_index)`

###	参数

####	shape类型

-	batch_size
	-	batch_size在实际数据输入中为首维（0维）
	-	shape类型参数传递的tuple中一般不包括batch_size维度
	-	输出时使用`None`表示`(batch_size,...)`

-	time_step
	-	对时序数据，time_step在实际数据输入中第二维（1维）

#####	`input_shape`

-	是`Layer`的初始化参数，所有`Layer`子类都具有

-	如果Layer是首层，需要传递该参数指明输入数据形状，否则
	无需传递该参数
	-	有些子类有类似于`input_dim`等参数具有`input_shape`
		部分功能

-	`None`：表示该维度变长

###	输入、输出

-	channels/depth/features：时间、空间单位上独立的数据，
	卷积应该在每个channal分别“独立”进行
	-	对1维时序（时间），channels就是每时刻的features
	-	对2维图片（空间），channels就是色彩通道
	-	对3维视频（时空），channels就是每帧色彩通道
	-	中间数据，channnels就是每个filters的输出

-	*1D*：`(batch, dim, channels)`（*channels_last*）

-	*2D*：`(batch, dim_1, dim_2, channels)`
	（*channels_last*）

-	*3D*：`(batch, dim_1, dim_2, dim_3, channels)`
	（*channels_last*）







