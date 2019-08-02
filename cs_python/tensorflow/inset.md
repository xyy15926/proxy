---
title: TensorFlow安装配置
tags:
  - Python
  - Tensorflow
categories:
  - Python
  - Tensorflow
date: 2019-03-21 17:27:37
updated: 2019-02-17 11:57:07
toc: true
mathjax: true
comments: true
description: TensorFlow安装配置
---

##	安装

##	TensorBoard

TensorBoard是包括在TensorFlow中可视化组件

-	运行启动了TB的TF项目时，操作都会输出为事件日志文件
-	TB能够把这些日志文件以可视化的方式展示出来
	-	模型图
	-	运行时行为、状态

```python
$ tensorboard --logdir=/path/to/logdir --port XXXX
```

##	问题

>	Your CPU supports instructions that this TensorFlow binary was not cmpiled to use: SSE1.4, SSE4.2, AVX AVX2 FMA

-	没从源代码安装以获取这些指令集的支持
	-	从源代码编译安装
	-	或者
		```python
		import os
		os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
		import tensorflow as tf
		```
