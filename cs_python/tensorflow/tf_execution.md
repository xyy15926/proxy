---
title: TensorFlow执行
tags:
 - Python
 - TensorFlow
categories:
 - Python
 - TensorFlow
 - Machine Learning
 - Execution Model
date: 2019-08-19 17:09
updated: 2019-08-19 17:09
toc: true
mathjax: true
comments: true
description: TensorFlow中Graph、Session、Eager Execution
---

##	Session

*Session*：TF中OPs对象执行、Tensor对象计算的封装环境

-	session管理图、OPs
	-	所有图必须Session中执行
	-	将图中OPs、其执行方法分发到CPU、GPU、TPU等设备上
	-	为当前变量值分配内存

-	session只执行*Data/Tensor Flow*中OPs，忽略不相关节点
	-	即通往`fetches`的flow中的**OPs构成子图**才会被计算

-	各session中值独立维护，不会相互影响

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

###	`config`参数选项

-	`log_device_placement`：打印每个操作所用设备
-	`allow_soft_placement`：允许不在GPU上执行操作自动迁移到
	CPU上执行
-	`gpu_options`
	-	`allow_growth`：让TF按需分配显存
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
	-	节省计算资源，可以只计算取得目标值过程中必须计算的值
	-	图可以被会被分割成子图（即极大连通分量），便于自动区分
	-	子图可以分发到不同的设备上分布式执行，即模型并行
	-	许多通用ML模型是通过有向图教学、可视化的

###	构建图

```python
class Graph:
	def __init__(self):
		pass

	# 将当前图作为默认图
	def as_default(self):
		pass

	# 强制OPs依赖关系（图中未反映），即优先执行指定OPs
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

###	Lazy Loading

*Lazy Loading*：推迟声明/初始化OP对象至载入图时
（不是指TF的延迟计算，是个人代码结构问题）

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

-	TF会默认占用所有GPU及每个GPU的所有显存
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

