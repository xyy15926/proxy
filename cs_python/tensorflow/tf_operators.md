---
title: TensorFlow操作符
tags:
 - Python
 - TensorFlow
categories:
 - Python
 - TensorFlow
 - Machine Learning
 - TensorFlow Operation
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
	-	TF的变量类型基于numpy变量类型；`tf.int32=tf.int32`
	-	大部分情况（bool、numeric）可以不加转换的使用TF、np
		变量类型
	-	对于string类型TF、np不完全一样，但TF仍然可以从numpy
		中导入string数组，但是需要注意不能在numpy中指定类型

-	尽量使用TF变量类型
	-	python原生类型：没有**细分**类型，TF需要推断类型
	-	numpy类型：numpy不兼容GPU，也不能自动计算衍生

-	`tf.float16`：16-bit half-precision floating-point
-	`tf.float32`：32-bit single-presicion floating-point
-	`tf.float64`：64-bit double-presicion floating-point
-	`tf.bfloat16`：16-bit truncated floating-point
-	`tf.complex64`：64-bit single-presicion complex
-	`tf.complex128`：128-bit double-presicion complex
-	`tf.int8`：8-bit signed integer
-	`tf.uint8`：8-bit unsigned integer
-	`tf.int16`：
-	`tf.uint16`
-	`tf.int32`
-	`tf.int64`
-	`tf.bool`
-	`tf.string`
-	`tf.qint8`：quantized 8-bit signed integer
-	`tf.quint8`
-	`tf.qint16`
-	`tf.quint16`
-	`tf.qint32`
-	`tf.resource`：handle to a mutable resource

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

-	`name`：重名时TF自动加上`_[num]`后缀

###	同值常量OPs

####	`tf.zeros`

```python
def zeros(
	shape,
	dtype=tf.float32,
	name=None
)
	# `np.zeros`
```

####	`tf.zores_like`

```python
def ones(
	shape,
	dtype=tf.float32,
	name=None
)
	# `np.ones`
```

####	`tf.zeros_like`

```python
def zeros_like(
	input_tensor,
	dtype=None,
	name=None,
	optimizeTrue
)
	# `np.zores_like`
```

####	`tf.ones_like`

```python
def ones_like(
	input_tensor,
	dtype=None,
	name=None,
	optimize=True
)
	# `np.ones_like`
```

-	`zeros_like`：若没有指明`dtype`，根据`input_tensor`确定
	-	对数值型为`0.0`
	-	对bool型为`False`
	-	对字符串为`b''`

-	`ones_like`：若没有指明`dtype`，根据`input_tensor`确定
	-	对数值型为`0.0`
	-	对bool型为`True`
	-	对字符串报错

####	`tf.fill`

-	以`value`填充`dims`给定形状
-	类似于`np.full`

```python
def fill(
	dims,
	value,
	name=None
)
```

###	列表常量OPs

-	tensor列表不能迭代

####	`tf.lin_space`

-	类似于`np.linspace`

```python
def lin_space(
	start,
	stop,
	num,
	name=None
)
```

####	`tf.range`

-	类似于`np.araneg`

```python
def range(
	start,
	limit=None,
	delta=1,
	dtype=None,
	name="range"
)
```

###	随机常量OPs

####	`tf.set_random_seed`

```python
def set_random_seed(seed)
```

####	`tf.random_normal`

####	`tf.truncated_normal`

####	`tf.random_uniform`

####	`tf.random_shuffle`

####	`tf.random_crop`

####	`tf.multinomial`

####	`tf.random_gamma`

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

####	函数

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

def reciprocal()
```

###	列表OPs

####	`tf.Concat`

####	`tf.Slice`

####	`tf.Split`

####	`tf.Rank`

####	`tf.Shape`

####	`tf.Shuffle`

###	矩阵OPs

####	`tf.MatMul`

####	`tf.MatrixInverse`

####	`tf.MatrixDeterminant`

####	`tf.tensordot`

矩阵点乘

###	Optimizer OPs

用于最小化loss的OPs

####	`tf.train.GradientDescentOptimizer`



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

	def intializer():
		pass
		# init OP

	def value():
		pass
		# read OP

	def assign():
		pass
		# write OP

	def assign_add():
		# 加和
```

-	在会话中**初始化**后，才能在会话中使用Variable
-	每个会话有自己的Variable副本，不相互影响

###	创建Variable

####	`tf.Variable`

此传统方法不再被提倡

```python
s = tf.Variable(2, name="scalar")
m = tf.Variable([[0,1], [2,3]], name="matrix")
w = tf.Variable(tf.zeros([784, 10]))
	# create Variable with tf.Variable
```

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

这个封装是被推荐的方法

-	allows for easy variable sharing
-	可以提供更多的参数
	-	`initailizer`：用于初始化值
	-	`trainable`：是否可训练

```python
s = tf.get_variable("scalar", initializer=tf.constant(2))
m = tf.get_variable("matrix", initializer=tf.constant([[0,1], [2,3]])
W = tf.get_variable("big_matrix",
	shape=(784, 10),
	initializer=tf.zeros_initializer()
)
	# 使用`get_variable`更好
```

###	初始化Variable

####	`tf.global_variable_initializer`

在图中初始化所有Variables

```python
with tf.Session() as sess:
	sess.run(tf.global_variable_initialier())
		# 初始化所有Variables
```

####	`tf.variable_initializer`

在图中初始化给定Variables

```python
with tf.Session() as sess:
	sess.run(tf.variable_initializer([s, m])
		# 初始化变量子集
```

####	`Variable.initializer`

初始化单个变量

```python
with tf.Session() as sess:
	sess.run(s.initializer)
		# 初始化单个变量
```

####	注意

-	如果一个Variable依赖另一个Variable，需要使用
	`initialized_value`确保依赖线性初始化
	```python
	W = tr.Variable(tf.truncated_normal([700, 100])
	U = tf.Variable(W.initialized_value() * 2)
	```

###	计算Variable

####	`Variable.eval`

除了`Session.run(Variable)`之外，还可以使用`.eval`

```python
with tf.Session() as sess:
	sess.run(s.initializer)
		# 初始化单个变量
	print(s.eval())
		# 计算变量值，类似于`sess.run(s)`
```

###	Variable值OPs

-	Variable的*方法*OPs和一般的OP一样，也需要在Session中执行
	才能生效

####	`Variable.assign`

-	`assgin`内部有初始化Variable，所以有时可以不用初始化
-	`assgin_add`、`assign_sub`方法并不会初始化Variable，因为
	其依赖于原始值

```python
assign_op = s.assign(100)
with tf.Session() as sess:
	sess.run(s.initialier)
	print(s.eval())
		# 打印`10`，assign OP未执行、不生效
	sess.run(assgin_op)
		# s=100
		# `assign`方法会隐式初始化Variable
	print(s.eval())
		# 打印`100`
	print(s.assign_add(10).eval())
		# 110
	print(s.assign_sub(20).eval())
		# 90
```

###	Variables VS Constants

-	constants是常数OPs
	-	存储在图中：每次载入图会同时被载入，过大的constants
		会使得载入图非常慢
	-	最好只对原生类型使用constants

-	Variables是TF对象，包括部分OPs
	-	和图分开存储，甚至是存储在独立的参数服务器上
	-	Variables存储大量数据也不会拖慢图载入速度
	-	通常用于存储训练过程中weight、bias


####	变量

-	变量维护图执行过程中状态信息

```python
state = tf.Variable(0, name="counter")
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)
init_op = tf.initialize_all_variables()
	# 启动图后，初始化op
	# 会报错？？？

withe tf.Session() as sess:
	sess.run(init_op)
	print(sess.run(state))
	for _ in range(3):
		sess.run(update)
		print(sess.run(state))
```

