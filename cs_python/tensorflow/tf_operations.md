---
title: TensorFlow操作符
tags:
 - Python
 - TensorFlow
 - Machine Learning
 - Operation
categories:
 - Python
 - TensorFlow
date: 2019-08-19 17:13
updated: 2019-08-19 17:13
toc: true
mathjax: true
comments: true
description: TensorFlow中操作符
---

##	Tensor

张量：n-dimensional array，类型化的多维数组

-	TF使用Tensor表示所有的数据
-	Tensor包含一个静态类型rank、一个shape
-	TF会将python原生类型转换为相应的Tensor
	-	0-d tensor：scalar
	-	1-d tensor：vector，1d-array
	-	2-d tensor：matrix，2d-array

###	Data Type

-	TF被设计为和numpy可以无缝结合
	-	TF的变量类型基于numpy变量类型：`tf.int32==np.int32`
	-	bool、numeric等大部分类型可以不加转换的使用TF、np
		变量类型
	-	TF、np中string类型不完全一样，但TF仍然可以从numpy
		中导入string数组，但是不能在numpy中指定类型

-	但尽量使用TF变量类型
	-	python原生类型：没有**细分**类型，TF需要推断类型
	-	numpy类型：numpy不兼容GPU，也不能自动计算衍生类型

|数据类型|说明|
|-----|-----|
|`tf.float16`|16-bit half-precision floating-point|
|`tf.float32`|32-bit single-presicion floating-point|
|`tf.float64`|64-bit double-presicion floating-point|
|`tf.bfloat16`|16-bit truncated floating-point|
|`tf.complex64`|64-bit single-presicion complex|
|`tf.complex128`|128-bit double-presicion complex|
|`tf.int8`|8-bit signed integer|
|`tf.uint8`|8-bit unsigned integer|
|`tf.int16`||
|`tf.uint16`||
|`tf.int32`||
|`tf.int64`||
|`tf.bool`||
|`tf.string`||
|`tf.qint8`|quantized 8-bit signed integer|
|`tf.quint8`||
|`tf.qint16`||
|`tf.quint16`||
|`tf.qint32`||
|`tf.resource`|handle to a mutable resource|

##	Constant OPs

###	`tf.constant`

```python
def constant(
	value,
	dtype=none,
	shape=none,
	name="Const",
	verify_shape=False
)
```

###	同值常量OPs

-	*zeros*：类似np中相应函数

	```python
	# `np.zeros`
	def tf.zeros(
		shape,
		dtype=tf.float32,
		name=None
	)

	# `np.zores_like`
	def tf.zeros_like(
		input_tensor,
		dtype=None,
		name=None,
		optimizeTrue
	)
	```

	-	若没有指明`dtype`，根据`input_tensor`确定其中值
		-	对数值型为`0.0`
		-	对bool型为`False`
		-	对字符串为`b''`

-	*ones*：类似np中相应函数


	```python
	# `np.ones`
	def tf.ones(
		shape,
		dtype=tf.float32,
		name=None
	)

	# `np.ones_like`
	def tf.ones_like(
		input_tensor,
		dtype=None,
		name=None,
		optimize=True
	)
	```

	-	若没有指明`dtype`，根据`input_tensor`确定
		-	对数值型为`0.0`
		-	对bool型为`True`
		-	对字符串报错

-	*fill*：以`value`填充`dims`给定形状

	```python
	# `np.fill`
	def tf.fill(
		dims,
		value,
		name=None
	)
	```

###	列表常量OPs

> - tensor列表不能直接`for`语句等迭代

-	`tf.lin_space`：`start`、`stop`直接均分为`num`部分

	```python
	# `np.linspace`
	def lin_space(
		start,
		stop,
		num,
		name=None
	)
	```

-	`tf.range`：`start`、`stop`间等间隔`delta`取值

	```python
	# `np.arange`
	def tf.range(
		start,
		limit=None,
		delta=1,
		dtype=None,
		name="range"
	)
	```

###	随机常量OPs

-	*seed*：设置随机数种子

	```python
	# np.random.seed
	def tf.set_random_seed(seed):
		pass
	```

-	*random*：随机生成函数

	```python
	def tf.random_normal()
	def tf.truncated_normal(
		?avg=0/int/(int),
		stddev=1.0/float,
		seed=None/int,
		name=None
	):
		pass

	def tf.random_uniform(
		shape(1d-arr),
		minval=0,
		maxval=None,
		dtype=tf.float32,
		seed=None/int,
		name=None/str
	):
		pass

	def tf.random_crop()

	def tf.multinomial()

	def tf.random_gamma()
	```

-	*shuffle*

	```python
	def tf.random_shuffle()
	```

##	运算OPs

###	元素OPs

####	四则运算

```python

def add(x, y, name=None)
def subtract(x, y, name=None)
def sub(x, y, name=None)
def multiply(x, y, name=None)
def mul(x, y, name=None)
	# 加、减、乘

def floordiv(x, y, name=None)
def floor_div(x, y, name=None)
def div(x, y, name=None)
def truncatediv(x, y, name=None)
	# 地板除

def divide(x, y, name=None)
def truediv(x, y, name=None)
	# 浮点除

def realdiv(x, y, name=None)
	# 实数除法，只能用于实数？

def add_n(input, name=None)
	# `input`：list-like，元素shape、type相同
	# 累加`input`中元素的值
```

####	逻辑运算

```python
def greater()
def less()
def equal()
```

####	数学函数

```python
def exp()
def log()
def square()
def round()
def sqrt()
def rsqrt()
def pow()
def abs()
def negative()
def sign()
def reciprocal()		# 倒数
```

###	列表运算OPs

```python
def tf.Concat()
def tf.Slice()
def tf.Split()
def tf.Rank()
def tf.Shape()
def tf.Shuffle()
```

###	矩阵OPs

```python
def tf.MatMul()
def tf.MatrixInverse()
def tf.MatrixDeterminant()
def tf.tensordot()				# 矩阵点乘
```

###	梯度OPs

```c
def tf.gradients(				# 求`y`对`[xs]`个元素偏导
	ys: tf.OPs,
	xs: tf.OPs/[tf.OPs],
	grad_ys=None,
	name=None
)
def tf.stop_gradient(
	input,
	name=None
)
def clip_by_value(
	t,
	clip_value_min,
	clip_value_max,
	name=None
)
def tf.clip_by_norm(
	t,
	clip_norm,
	axes=None,
	name=None
)
```

##	Variable

```python
class Variable:
	def __init__(self,
		init_value=None,
		trainable=True,
		collections=None,
		validata_shape=True,
		caching_device=None,
		name=None,
		variable_def=None,
		dtype=None,
		expected_shape=None,
		import_scope=None,
		constraint=None
	)

	# 初始化变量
	# `sess.run`其即初始化变量
	def intializer(self):
		pass

	# 读取变量值
	def value(self):
		pass

	# 获取变量初始化值，其他变量调用、声明依赖该变量
	def initilized_value(self):
		pass

	# 计算、获取变量值，类似`sess.run(OP)`
	def eval(self):
		pass

	# 给变量赋值
	# `assgin`内部有初始化Variable，所以有时可以不用初始化
	def assign(self):
		pass
	#`assgin_add`等依赖于原始值，不会初始化变量
	def assign_add(self, ?dec)
	def assign_divide(self, ?dec)
```

-	`Variable`是包含很多方法的类
	-	其中**方法OPs**和一般的OP一样，也需要在Session中执行
		才能生效
	-	`Variable`必须在会话中**初始化**后，才能使用
	-	会话维护自身独立`Variable`副本，不相互影响

-	`Variable`和图分开存储，甚至是存储在独立参数服务器上
	-	存储大量数据也不会拖慢图载入速度
	-	通常用于存储训练过程中weight、bias、维护图执行过程
		中状态信息

-	constants是常数OPs
	-	存储在图中：每次载入图会同时被载入，过大的constants
		会使得载入图非常慢
	-	所以最好只对原生类型使用constants

###	Variable创建

####	`tf.get_variable`

```python
def get_variable(
	name,
	shape=None,
	dtype=None,
	initializer=None,
	regularizer=None,
	trainable=True,
	collections=None,
	caching_device=None,
	partitioner=None,
	validate_shape=True,
	use_resource=None,
	custom_getter=None,
	constraint=None
)
```

-	此封装工厂方法相较于直接通过`tf.Variable`更好
	-	若变量已设置，可通过变量名获取变量，方便变量共享
	-	可以提供更多的参数定制变量值

```python
	# `tf.Variable`创建变量
s = tf.Variable(2, name="scalar")
m = tf.Variable([[0,1], [2,3]], name="matrix")
w = tf.Variable(tf.zeros([784, 10]))
	# `tf.get_variable`创建、获取变量
s = tf.get_variable("scalar", initializer=tf.constant(2))
m = tf.get_variable("matrix", initializer=tf.constant([[0,1], [2,3]])
W = tf.get_variable("big_matrix",
	shape=(784, 10),
	initializer=tf.zeros_initializer()
)
```

###	Variable初始化

```python
with tf.Session() as sess:
	# 初始化所有Variables
	sess.run(tf.global_variable_initialier())
	# 初始化变量子集
	sess.run(tf.variable_initializer([s, m])
	# 初始化指定单个变量
	sess.run(s.initializer)
```

-	若某Variable依赖其他Variable，需要使用
	`initialized_value`指明依赖，确保依赖线性初始化

	```python
	W = tr.Variable(tf.truncated_normal([700, 100])
	# 指明依赖，保证依赖线性初始化
	U = tf.Variable(W.initialized_value() * 2)
	```

