#	TensorFlow内容

-	TF中所有对象都是Operators，需要执行才能得到Tensor作为
	结果
	-	即不能直接对他们**值**进行访问、比较、操作

##	Session

OPs对象执行、Tensor对象计算的封装环境

-	Session管理图、OPs
	-	所有图必须Session中执行
	-	将图中OPs、其执行方法分发到CPU、GPU、TPU等设备上
	-	为当前变量值分配内存
-	Session只执行flow中的OPs，不计算不需要的值
-	Session独立的维护其中的值

```python
tf.Session(
	target="",
	graph=None,
	config=None
)
```

-	`target`：执行引擎
-	`graph`：在Session中加载图，若没有指定参数则使用默认图
-	`config`：包含Session配置的protocal buffer

###	创建Session

####	`tf.Session`

```python
sess = tf.Session()
	# 创建会话
sess.close()
	# 关闭会话以释放资源

with tf.Session() as sess:
	pass
	# 由`with`管理Session开、关
```

####	`tf.InteractiveSession`

```python
sess = tf.InteractiveSession()
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
print(c.eval())
	# 开启交互式Session后，可以不需要显式在Session中执行
		# 即不需要在在`sess.run()`中执行
sess.close()
```

###	`Session.run`

```python
tf.Session.run(fetches,
	feed_dict=None,
	options=None,
	run_metadata=None)
```

-	`fetches`：需要获得/计算的OPs值列表
	-	值若为Tensor
		-	python中：以`np.ndarray`返回
		-	C++/C中：以`tensorflow:Tensor`返回
	-	通往`fetches`的flow中的OPs构成子图才会被计算

-	`feed_dict`：传递给Variables的feed值字典，替换图中Tensor
	（常配合`tf.placeholder`一起使用）
	-	`{var_name}: val`：`var_name`为被替换值的变量名
	-	被替换的可以是任何Tensor，包括常量OPs产生的Tensor

-	如果只需要得到某个OP值可以直接调用`.eval()`获得

##	Graph、Operation

在TF表示计算任务

-	图由Operations构成
-	OPs获得一个、多个Tensor用于执行计算，产生、返回Tensor
-	图可能会被分割成小块，分发到不同的设备上并行执行

###	构建图

-	TensorFlow库中有一个默认图
	-	OP构造器默认为其增加节点
	-	`tf.Session`默认也是执行默认图

-	可以通过`tf.Graph`创建新图，但是最好在是在一张图中使用
	多个不相连的子图，而不是多张图
	-	充分性：Session执行图时忽略不必要的OPs
	-	必要性
		-	多张图需要多个会话，每张图执行时默认尝试使用所有
			可能资源
		-	不能通过python/numpy在图间传递数据（分布式系统）

-	使用图的好处
	-	节省计算：可以只计算取得目标值过程中必须计算的值
	-	将计算拆分成小的、不同的块便于自动区分
	-	利于将分布式计算，方便将任务发布在多个设备上
	-	许多通用ML模型是通过有向图教学、可视化的

```python
g1 = tf.get_default_graph()
	# to handle the default graph

g2 = tf.Graph()
	# create a graph

with g1.as_default():
	a = tf.Constant(3, name="a")
	# `g1`本来就是默认图，这个`with`块仅仅是为了区分

with g2.as_default():
	b = tf.Constant(4)
	# 向用户自定义图`g2`添加OPs
```

###	计算控制

####	`Graph.control_dependencies`

-	指定优先执行某些OPs，即使这些OPs不相互依赖

```python
with g.control_dependencies([a, b, c]):
	pass
```

###	可视化

####	`Graph.as_graph_def`

-	以protocal buffer的形式展示图（类似JSON）

>	protocal buffer: google's language-neutral, platform-neutral,
	extensible mechanism for serializing structured data-think XML,
	but smaller, faster, and simpler

```python
print(g1.as_graph_def())
```

####	`tf.summary.FileWriter`

-	创建`FileWriter`对象用于记录log
-	存储图到**文件夹**中，文件名由TF自行生成
-	生成event log文件可以通过TensorBoard组件查看

```python
writer = tf.summary.FileWriter("./graphs", g1)
	# 创建`FileWriter`用于记录log
	# 在图定义/构建完成后、会话**执行**图前创建
with tf.Session() as sess:
	# writer = tf.summary.FileWriter("./graphs", sess.graph)
		# 也可以在创建Session之后，记录Session中的图
	session.run(a)
write.close()
	# 关闭`FileWriter`，生成event log文件
```

###	其他函数

-	`Graph.is_feedable`：判断图中该OPs是否可以被替换

###	注意

####	Lazy Loading

TF会推迟声明/初始化对象直到开始载入，这就会导致问题

-	正常代码
	```python
	x = tf.Variable(10, name='x')
	y = tf.Variable(20, name='y')
	z = tf.add(x, y)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		writer = tf.summary.FileWriter('graphs/normal_loading', sess.graph)
		for _ in range(10):
			sess.run(z)
		writer.close()
	```

-	“有问题”代码
	```python
	x = tf.Variable(10, name='x')
	y = tf.Variable(20, name='y')

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		writer = tf.summary.FileWriter('graphs/lazy_loading', sess.graph)
		for _ in range(10):
			sess.run(tf.add(x, y))
			# 这里每执行一次都会增加无必要的`add` OP
			# 可能会影响图的载入、传递
		print(tf.get_default_graph().as_graph_def())
		writer.close()
	```

-	解决方案
	-	总是把（图的搭建）OPs的定义、执行分开
	-	否则利用`@property`，使用函数封装变量进行控制，保证
		只在第一次访问时才创建

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

##	控制OPs

###	Neural Network Building Blocks

####	`tf.softmax`

####	`tf.Sigmod`

####	`tf.ReLU`

####	`tf.Convolution2D`

####	`tf.MaxPool`

###	Checkpointing

####	`tf.Save`

####	`tf.Restore`

###	Queue and Synchronization

####	`tf.Enqueue`

####	`tf.Dequeue`

####	`tf.MutexAcquire`

####	`tf.MutexRelease`

###	Control Flow

####	`tf.count_up_to`

####	`tf.cond`

`pred`为`True`，执行`true_fn`，否则执行`false_fn`

```python
tf.cond(
	pred,
	true_fn=None,
	false_fn =None,
)
```

####	`tf.case`

####	`tf.while_loop`

####	`tf.group`

####	`tf.Merge`

####	`tf.Switch`

####	`tf.Enter`

####	`tf.Leave`

####	`tf.NextIteration`

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

##	Device

###	设备标识

设备使用字符串进行标识

-	`/cpu:0`：所有CPU都以此作为名称
-	`/gpu:0`：第一个GPU，如果有
-	`/gpu:1`：第二个GPU

###	设备执行

-	TF将图形定义转换成分布式执行的操作以充分利用计算资源，
	一般无需显式指定使用CPU、GPU，TF能自动检测，尽可能使用
	找到的第一个GPU执行操作

-	TF会默认占用所有GPU及每个GPU的所有显存
	-	但只有一个GPU参与计算，其他GPU默认不参与计算（显存
		仍然被占用），需要明确将OPs指派给其执行
	-	指派GPU数量需通过设置环境变量实现
	-	控制显存占用需设置Session `config`参数

-	注意：有些操作不能再GPU上完成，手动指派计算设备需要注意

###	环境变量

同样既可以修改`os.environ`，也可以在shell中设置环境变量

-	`CUDA_VISIBLE_DEVICES`：可被使用的GPU id

###	会话参数

配置Session `config`参数对象

-	`log_device_placement`：打印每个操作所用设备
-	`allow_soft_placement`：允许不在GPU上执行操作自动迁移到
	CPU上执行
-	`gpu_options`
	-	`allow_growth`：让TF按需分配显存
	-	`per_process_gpu_memory_fraction`：指定每个GPU进程
		使用显存比例（无法对单个GPU分别设置）

###	示例

```python
conf = tf.ConfigProto(log_device_placement=True)
conf.gpu_options.allow_growth=True
with tf.device("/cpu:2"):
	a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0], name="a")
	b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0], name="b")
	c = tf.multiply(a, b)
	# creates a graph

sess = tf.Session(config=conf)
	# 这样程序会显示每个操作所用的设备、按需使用GPU显存

print(sess.run(c))
	# runs the op.

sess.close()
```

##	Resources

###	`tf.placeholder`

占位符，之后在真正`.run`时，通过`feed_dict`参数设置值

-	`shape`：并不必须，但最好设置参数方便debug
-	需要导入数据给placeholder，可能影响程序速度

```python
tf.placeholder(
	dtype,
	shape=None,
	name=None
)
```

###	`tf.data`

直接将数据存储在`tf.data.Dataset`对象中

####	`tf.data.Dataset.from_tensor_slice`

-	`features`、`labels`：Tensors或者是ndarray

```python
from_tensor_slice(
	(features, labels)
)
```

####	文件中读取数据

可以从多个文件中读取数据

```python
tf.data.TextLineDataset(filenames)
	# 文件每行为一个entry
tf.data.FixedLengthRecordDataset(filenames)
	# 文件中entry定长
tf.data.TRRecordDataset(filenames)
	# 文件为tfrecord格式
```

####	使用数据

```python
iterator = dataset.make_one_shot_iterator()
	# 创建只能迭代一轮的迭代器
X, Y = iterator.get_next()
	# 这里`X`、`Y`也是OPs，在执行时候才返回Tensor

with tf.Session() as sess:
	print(sess.run([X, Y]))
	print(sess.run([X, Y]))
	print(sess.run([X, Y]))
		# 每次不同的值

iterator = data.make_initializable_iterator()
	# 创建可以多次初始化的迭代器
with tf.Session() as sess:
	for _ in range(100):
		sess.run(iterator.initializer)
			# 每轮重新初始化迭代器，重新使用
		total_loss = 0
		try:
			while True:
				sess.run([optimizer])
		except tf.error.OutOfRangeError:
			# 手动处理迭代器消耗完毕
			pass
```

####	处理数据

```python
dataset = dataset.shuffle(1000)
dataset = dataset.repeat(1000)
dataset = dataset.batch(128)
dataset = dataset.map(lambda x: tf.one_hot(x, 10))
```


