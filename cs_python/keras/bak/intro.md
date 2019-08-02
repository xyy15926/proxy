---
title: Keras介绍
tags:
  - Python
  - Keras
categories:
  - Python
  - Keras
date: 2019-02-20 23:58:15
updated: 2019-02-17 11:57:07
toc: true
mathjax: true
comments: true
description: Keras介绍
---

##	Keras后端

Keras是一个模型级的库，提供了快速构建深度学习网络的模块

-	并不处理如张量乘法、卷积等底层操作
-	这些操作依赖于某种特定的、优化良好的张量操作库
-	被依赖的处理张量的库就称为*后端引擎*
	-	Theano：开源的符号主义张量操作框架，由蒙特利尔大学
		LISA/MILA实验室开发
	-	Tensorflow：符号主义的张量操作框架，由Google开发
	-	CNTK：微软开发的商业级工具包
-	Keras将其函数统一封装，使得用户可以以同一个接口调用不同	
	后端引擎的函数

###	切换后端

-	修改Keras配置文件
-	定义环境变量`KERAS_BACKEND`覆盖配置文件中设置（见python
	修改环境变量的3中方式）

###	Keras后端抽象

可以通过Keras后端接口来编写代码，使得Keras模块能够同时在
Theano和TensorFlow两个后端上使用

-	大多数张量操作都可以通过统一的Keras后端接口完成，不必
	关心具体执行后端

```python
from keras import backend as K

input = K.placeholder(shape=(2, 4, 5))
input = K.placeholder(shape=(None, 4, 5))
input = K.placeholder(ndim=3)
	# 实例化输出占位符

val = np.random.random((3, 4, 5))
var = K.variable(value=val)
	# 实例化共享变量
	# 等价于`tf.Variable`、`theano.shared`
var = K.zeros(shape=(3, 4, 5))
var = K.ones(shape=(3, 4, 5))
```

###	后端函数

####	`K.backend`

####	配置

-	`epsilon`
-	`set_epsilon`
-	`floatx`
-	`set_floatx`
-	`image_data_format`
-	`set_iamge_data_format`
-	`get_uid`:

####	`

-	`cast_to_floatx`：将NDA转换为默认Kersas floatx类型


##	Keras配置

###	`$HOME/.keras/keras.json`

```json
{
	"image_data_format": "channel_last",
		# 指定Keras将要使用数据维度顺序
	"epsilon": 1e-07,
		# 防止除0错误数字
	"flaotx": "float32",
		# 浮点数精度
	"backend": "tensorflow"
		# 指定Keras所使用后端
}
```





