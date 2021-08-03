---
title: TensorFlow执行
categories:
 - Python
 - TensorFlow
tags:
 - Python
 - TensorFlow
 - Machine Learning
 - Execution Model
date: 2019-08-19 17:09
updated: 2021-08-02 15:07:32
toc: true
mathjax: true
comments: true
description: TensorFlow中Graph、Session、Eager Execution
---

##	Session

*Session*：TF中OPs对象执行、Tensor对象计算的封装环境

-	Session管理图、OPs
	-	所有图必须Session中执行
	-	将图中OPs、其执行方法分发到CPU、GPU、TPU等设备上
	-	为当前变量值分配内存

-	Session只执行*Data/Tensor Flow*中OPs，忽略不相关节点
	-	即通往`fetches`的flow中的**OPs构成子图**才会被计算

-	各Session中各值独立维护，不会相互影响

```python
class Session:
	def __init__(self,
		target=str,
		graph=None/tf.Graph,
		config=None/tf.ConfigProto)

		# 获取Session中载入图
		self.graph

	# 关闭会话，释放资源
	def close(self):
		pass

	# 支持上下文语法
	def __enter__(self):
		pass
	def __exit__(self):
		pass

	# 执行TensorFlow中节点，获取`fetches`中节点值
	# 返回值若为Tensor
		# python中：以`np.ndarray`返回
		# C++/C中：以`tensorflow:Tensor`返回
	# 可直接调用`.eval()`获得单个OP值
	def run(self,
		fetches=tf.OPs/[tf.OPs],
		feed_dict=None/dict,
		options=None,
		run_metadata=None)
```

###	`tf.InteractiveSession`

`tf.InteractiveSession`：开启交互式会话

```python
 # 开启交互式Session
sess = tf.InteractiveSession()
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
x = tf.Variable([1.0, 2.0])
 # 无需显式在`sess.run`中执行
 # 直接调用`OPs.eval/run()`方法得到结果
x.initializer.run()
print(c.eval())
sess.close()
```

###	Session执行

-	Fetch机制：`sess.run()`执行图时，传入需**取回**的结果，
	取回操作输出内容

-	Feed机制：通过`feed_dict`参数，使用自定义tensor值替代图
	中任意feeable OPs的输出

	-	`tf.placeholder()`表示创建占位符，执行Graph时必须
		使用tensor替代
	-	`feed_dict`只是函数参数，只在调用它的方法内有效，
		方法执行完毕则消失

-	可以通过`feed_dict` feed所有feedable tensor，placeholder
	只是指明必须给某些提供值

	```python
	a = tf.add(2, 5)
	b = tf.multiply(a, 3)
	with tf.Session() as sess:
		sess.run(b, feed_dict={a: 15})			# 45
	```

###	`config`参数选项

-	`log_device_placement`：打印每个操作所用设备
-	`allow_soft_placement`：允许不在GPU上执行操作自动迁移到
	CPU上执行
-	`gpu_options`
	-	`allow_growth`：按需分配显存
	-	`per_process_gpu_memory_fraction`：指定每个GPU进程
		使用显存比例（无法对单个GPU分别设置）

> - 具体配置参见`tf.ConfigProto`

##	Graph

*Graph*：表示TF中计算任务，

-	*operation/node*：Graph中节点，包括：*operator*、
	*variable*、*constant*
	-	获取一个、多个Tensor执行计算、产生、返回tensor
	-	不能无法对其值直接进行访问、比较、操作
	-	图中节点可以命名，TF会自动给未命名节点命名

-	*tensor*：Graph中边，n维数组
	-	TF中所有对象都是Operators
	-	tensor是OPs执行结果，在图中传递/流动

-	图计算模型优势
	-	优化能力强、节省计算资源
		-	缓冲自动重用
		-	常量折叠，只计算取得目标值过程中必须计算的值
		-	方便并行化
		-	自动权衡计算、存储效率
	-	易于部署
		-	可被割成子图（即极大连通分量），便于自动区分
		-	子图可以分发到不同的设备上分布式执行，即模型并行
		-	许多通用ML模型是通过有向图教学、可视化的

-	图计算模型劣势
	-	难以debug
		-	图定义之后执行才会报错
		-	无法通过pdb、打印状态debug
	-	语法繁复

###	构建图

```python
class Graph:
	def __init__(self):
		pass

	# 将当前图作为默认图
	# 支持上下文语法
	def as_default(self):
		pass

	# 强制OPs依赖关系（图中未反映），即优先执行指定OPs
	# 支持上下文语法
	def as_default(self):
	def control_dependencies(self,
		?ops=[tf.OPs])
		pass

	# 以ProtoBuf格式展示Graph
	def as_graph_def(self):
		pass

	# 判断图中节点是否可以被feed
	def is_feedable(self,
		?op=tf.OPs):
		pass
```

-	可以通过`tf.Graph`创建新图，但最好在是在一张图中使用多个
	不相连的子图，而不是多张图
	-	充分性：Session执行图时忽略不必要的OPs
	-	必要性
		-	多张图需要多个会话，每张图执行时默认尝试使用所有
			可能资源
		-	不能通过python/numpy在图间传递数据（分布式系统）

-	初始化即包含默认图，OP构造器默认为其增加节点
	-	通过`tf.get_default_graph()`获取

###	图相关方法

-	获取TF初始化的默认图

	```python
	def tf.get_default_graph():
		pass
	```

###	命名空间

```python
 # 均支持、利用上下文语法，将OPs定义于其下
def tf.name_scope(name(str)):
	pass
def tf.variable_scope(
	name(str),
	reuse=tf.AUTO_REUSE
):
	pass
```
-	`tf.name_scope`：将变量分组
	-	只是将变量打包，不影响变量的重用、可见性等
	-	方便管理、查看graph
-	`tf.variable_scope`：
	-	对变量有控制能力
		-	可设置变量重用性
		-	变量可见性局限于该*variable scope*内，即不同
			variable scope间可以有完全同名变量
			（未被TF添加顺序后缀）
	-	会隐式创建*name scope*
	-	大部分情况是用于实现变量共享

```python
def fully_connected(x, output_dim, scope):
	# 设置variable scope中变量自动重用
	# 或者调用`scope.reuse_variables()`声明变量可重用
	with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
		# 在当前variable scope中获取、创建变量
		w = tf.get_variable("weights", [x.shape[1]], output_dim,
							initializer=tf.random_normal_initializer())
		b = tf.get_variable("biases", [output_dim],
							initializer=tf.constant_initizer(0.0))
		return tf.matmul(x, w) + b

def two_hidden_layers(x):
	h1 = fully_connected(x, 50, "h1")
	h2 =-fully_connected(h1, 10, "h2")

with tf.variable_scope("two_layers") as scope:
	logits1 = two_hidden_layers(x1)
	logits2 = two_hidden_layers(x2)
```

###	Lazy Loading

*Lazy Loading*：推迟声明/初始化OP对象至载入图时
（不是指TF的延迟计算，是个人代码结构问题，虽然是TF延迟图计算
模型的结果）

-	延迟加载容易导致向图中添加大量重复节点，影响图的载入、
	传递

	```python
	x = tf.Variable(10, name='x')
	y = tf.Variable(20, name='y')

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		writer = tf.summary.FileWriter('graphs/lazy_loading', sess.graph)
		for _ in range(10):
			# 延迟加载节点，每次执行都会添加`tf.add`OP
			sess.run(tf.add(x, y))
		print(tf.get_default_graph().as_graph_def())
		writer.close()
	```

-	解决方案
	-	总是把（图的搭建）OPs的定义、执行分开
	-	python利用`@property`装饰器，使用**单例模式函数**
		封装变量控制，保证仅首次调用函数时才创建OP

##	Eager Execution

```python
import tensorflow as tf
import tensorflow.contrib.eager as tfe
 # 启用TF eager execution
tfe.enable_eager_execution()
```

-	优势
	-	支持python debug工具
	-	提供实时报错
	-	支持python数据结构
	-	支持pythonic的控制流

		```python
		i = tf.constant(0)
		whlile i < 1000:
			i = tf.add(i, 1)
		```

-	eager execution开启后
	-	tensors行为类似`np.ndarray`
	-	大部分API和未开启同样工作，倾向于使用
		-	`tfe.Variable`
		-	`tf.contrib.summary`
		-	`tfe.Iterator`
		-	`tfe.py_func`
		-	面向对象的layers
		-	需要自行管理变量存储

-	eager execution和graph大部分兼容
	-	checkpoint兼容
	-	代码可以同时用于python过程、构建图
	-	可使用`@tfe.function`将计算编译为图

###	示例

-	placeholder、sessions

	```python
	# 普通TF
	x = tf.placholder(tf.float32, shape=[1, 1])
	m = tf.matmul(x, x)
	with tf.Session() as sess:
		m_out = sess.run(m, feed_dict={x: [[2.]]})

	# Eager Execution
	x = [[2.]]
	m = tf.matmul(x, x)
	```

-	Lazy loading

	```python
	x = tf.random_uniform([2, 2])
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			# 不会添加多个节点
			print(x[i, j])
	```

##	Device

###	设备标识

-	设备标识：设备使用字符串进行标识
	-	`/cpu:0`：所有CPU都以此作为名称
	-	`/gpu:0`：第一个GPU，如果有
	-	`/gpu:1`：第二个GPU

	```python
	# 为计算指定硬件资源
	with tf.device("/gpu:2"):
		a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0], name="a")
		b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0], name="b")
		c = tf.multiply(a, b)
		# creates a graph
	```

-	环境变量：python中既可以修改`os.environ`，也可以直接设置
	中设置环境变量
	-	`CUDA_VISIBLE_DEVICES`：可被使用的GPU id

###	设备执行

设备执行：TF将图形定义转换成分布式执行的操作以充分利用计算
资源

-	TF默认自动检测硬件配置，尽可能使用找到首个GPU执行操作

-	TF会默认占用所有GPU及每个GPU的所有显存（可配置）
	-	但只有一个GPU参与计算，其他GPU默认不参与计算（显存
		仍然被占用），需要明确将OPs指派给其执行
	-	指派GPU数量需通过设置环境变量实现
	-	控制显存占用需设置Session `config`参数

> - 注意：有些操作不能再GPU上完成，手动指派计算设备需要注意

##	`tf.ConfigProto`

-	Session参数配置

	```python
	 # `tf.ConfigProto`配置方法
	conf = tf.ConfigProto(log_device_placement=True)
	conf.gpu_options.allow_growth=True
	sess = tf.Session(config=conf)
	sess.close()
	```

