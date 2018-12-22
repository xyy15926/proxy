#	Keras

##	Keras后端

Keras是一个模型级的库，提供了快速构建深度学习网络的模块

-	Keras并不处理如张量乘法、卷积等底层操作，而是依赖于某种
	特定的、优化良好的张量操作库

-	Keras依赖于处理张量的库就称为“后端引擎”

	-	[Theano][Theano]：开源的符号主义张量操作框架，由
		蒙特利尔大学LISA/MILA实验室开发
	-	[Tensorflow][Tensorflow]：符号主义的张量操作框架，
		由Google开发
	-	[CNTK][CNTK]：由微软开发的商业级工具包

-	Keras将其函数统一封装，使得用户可以以同一个接口调用不同
	后端引擎的函数

* [Theano]: http://deeplearning.net/software/theano/
* [TensorFlow]: http://www.tensorflow.org/
* [CNTK]: https://www.microsoft.com/en-us/cognitive-toolkit/

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

## Kera后端函数

###	配置相关

#### `backend`

返回当前后端

####	`epsilon`

数值表达式中使用fuzz factor，用于防止除0错误

####	`set_epsilon`

设置*fuzz factor*

####	`floatx`

返回默认的浮点数数据类型

-	`float16`
-	`float32`
-	`float64`

####	`set_floatx`

设置默认的浮点数数据类型（字符串表示）

-	'float16'
-	'float32'
-	'float64'

####	`cast_to_floatx`

将NDA转换为默认的Keras floatx类型（的NDA）

####	`image_data_format`

返回图像的默认维度顺序

-	`channels_last`
-	`channels_first`

####	`set_image_data_format`

设置图像的默认维度顺序

-	`tf`
-	`th`

###	运算相关

####	`is_keras_tensor`

判断是否是Keras Tensor对象

```python
from keras import backend as K
np_var = numpy.array([1, 2])
K.is_keras_tensor(np_var)
	# False
keras_var = K.variable(np_var)
K.is_keras_tensor(keras_var)
	# A variable is not a Tensor.
	# False
keras_placeholder = K.placeholder(shape=(2, 4, 5))
K.is_keras_tensor(keras_placeholder)
	# A placeholder is a Tensor.
	# True
```

####	`get_uid`

获得默认计算图的uid，依据给定的前缀提供一个唯一的UID

-	参数：为表示前缀的字符串
-	返回值：整数

####	`reset_uids`

重置图的标识符

####	`clear_session`

结束当前的TF计算图，并新建一个

-	有效的避免模型/层的混乱

####	`manual_variable_initialization`

设置变量应该以其默认值被初始化还是由用户手动初始化

-	参数：布尔值，默认False代表变量由其默认值初始化

####	`learning_phase`

返回训练模式/测试模式的flag

-	训练模式：`0`
-	测试模式：`1`

####	`set_learning_phase`

设置训练模式/测试模式

###	OPs、Tensors

####	`is_sparse`

判断一个Tensor是不是一个稀疏的Tensor

-	返回值：布尔值

>	稀不稀疏由Tensor的类型决定，而不是Tensor实际上有多稀疏

```python
from keras import backend as K
a = K.placeholder((2, 2), sparse=False)
print(K.is_sparse(a))
	# False
b = K.placeholder((2, 2), sparse=True)
print(K.is_sparse(b))
	# True
```

####	`to_dense`

将一个稀疏tensor转换一个不稀疏的tensor并返回

####	`variable`

```python
def variable(
	value,
	dtype='float32'/str,
	name=None
)
```
实例化一个张量，返回之

-	参数
	-	`value`：用来初始化张量的值
	-	`dtype`：张量数据类型
	-	`name`：张量的名字（可选）

####	`placeholder`

```python
def placeholder(
	shape=None,
	ndim=None,
	dtype='float32'/str,
	name=None
)
```

实例化一个占位符

-	参数

	-	`shape`：占位符的shape（整数tuple，可能包含None）
	-	`ndim`: 占位符张量的阶数
		-	要初始化占位符，至少指定`shape`和`ndim`之一，
			如果都指定则使用```shape```
	-	`dtype`: 占位符数据类型
	-	`name`: 占位符名称（可选）

####	`shape`

```python
def shape(x)
```

返回张量的*符号shape*

-	返回值：OPs（需要执行才能得到值）

```python
from keras import backend as K
tf_session = K.get_session()
val = np.array([[1, 2], [3, 4]])
kvar = K.variable(value=val)
input = keras.backend.placeholder(shape=(2, 4, 5))
K.shape(kvar)
	# <tf.Tensor 'Shape_8:0' shape=(2,) dtype=int32>
K.shape(input)
	# <tf.Tensor 'Shape_9:0' shape=(3,) dtype=int32>
	# To get integer shape (Instead, you can use K.int_shape(x))
K.shape(kvar).eval(session=tf_session)
	# array([2, 2], dtype=int32)
K.shape(input).eval(session=tf_session)
	# array([2, 4, 5], dtype=int32)
```

####	`int_shape`

```python
def int_shape(x)
```

返回张量shape

-	返回值：`tuple(int)/None`

```python
from keras import backend as K
input = K.placeholder(shape=(2, 4, 5))
K.int_shape(input)
	# (2, 4, 5)
val = np.array([[1, 2], [3, 4]])
kvar = K.variable(value=val)
K.int_shape(kvar)
	# (2, 2)
```

####	`ndim`

```python
def ndim(x)
```

返回张量的阶数

-	返回值：int

```python
from keras import backend as K
input = K.placeholder(shape=(2, 4, 5))
val = np.array([[1, 2], [3, 4]])
kvar = K.variable(value=val)
K.ndim(input)
	# 3
K.ndim(kvar)
	# 2
```

####	`dtype`

```python
def dtype(x)
```

返回张量的数据类型

-	返回值：str
	-	`float32`
	-	`float32_ref`

```python
from keras import backend as K
K.dtype(K.placeholder(shape=(2,4,5)))
	# 'float32'
K.dtype(K.placeholder(shape=(2,4,5), dtype='float32'))
	# 'float32'
K.dtype(K.placeholder(shape=(2,4,5), dtype='float64'))
	# 'float64'__Keras variable__

kvar = K.variable(np.array([[1, 2], [3, 4]]))
K.dtype(kvar)
	# 'float32_ref'
kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
K.dtype(kvar)
	# 'float32_ref'
```

#### `eval`

```python
def eval(x)
```

求得张量的值

-	返回值：NDA

```python
from keras import backend as K
kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
K.eval(kvar)
	# array([[ 1.,  2.],
	#	[ 3.,  4.]], dtype=float32)
```

####	`zeros

```python
def zeros(
	shape,
	dtype='float32',
	name=None
)
```

生成`shape`大小的全0张量

```python
from keras import backend as K
kvar = K.zeros((3,4))
K.eval(kvar)
	# array([[ 0.,  0.,  0.,  0.],
	#	[ 0.,  0.,  0.,  0.],
	#	[ 0.,  0.,  0.,  0.]], dtype=float32)
```

####	`ones`

```python
def ones(
	shape,
	dtype='float32',
	name=None
)
```

生成`shape`大小的全1张量

####	`eye`

```python
def eye(
	size,
	dtype='float32',
	name=None
)
```

生成`size`大小的单位阵

####	`zeros_like`

```python
def zeros_like(
	x,
	name=None
)
```

生成与`x` shape相同的全0张量

####	`ones_like`

```python
def ones_like(
	x,
	name=None
)
```

生成与`x` shape相同的全1张量

###	随机常量OPs

####	`random_uniform_variable`

```python
def random_uniform_variable(
	shape,
	low,
	high,
	dtype=None,
	name=None,
	seed=None
)
```

初始化均匀分布常量OPs

-	参数
	-	`low`：浮点数，均匀分布之下界
	-	`high`：浮点数，均匀分布之上界
	-	`dtype`：数据类型
	-	`name`：张量名
	-	`seed`：随机数种子

####	`count_params`

```python
def count_params(x)
```

返回张量中标量的个数

####	`cast`

```python
def cast(x, dtype)
```

改变张量的数据类型

-	参数
	-	`dtype`：`float16`/`float32`/`float64`

####	`update`

```python
def update(x, new_x)
```

用`new_x`更新`x`

####	`update_add`

```python
def update_add(x, increment)
```

将`x`增加`increment`并更新`x`

####	`update_sub`

```python
def update_sub(x, decrement)
```

将`x`减少`decrement`并更新`x`

####	`moving_average_update`

```python
def moving_average_update(
	x,
	value,
	momentum
)
```

使用移动平均更新`x`

####	`dot`

```python
def dot(x, y)
```

求两个张量的点乘

```python
x = K.placeholder(shape=(2, 3))
y = K.placeholder(shape=(3, 4))
xy = K.dot(x, y)
xy
	# <tf.Tensor 'MatMul_9:0' shape=(2, 4) dtype=float32>
```

-	当试图计算两个N阶张量的乘积时，与Theano行为相同
	`(2, 3).(4, 3, 5) = (2, 4, 5))`

	```python
	x = K.placeholder(shape=(32, 28, 3))
	y = K.placeholder(shape=(3, 4))
	xy = K.dot(x, y)
	xy
		# <tf.Tensor 'MatMul_9:0' shape=(32, 28, 4) dtype=float32>
	```

	```python
	x = K.random_uniform_variable(shape=(2, 3), low=0, high=1)
	y = K.ones((4, 3, 5))
	xy = K.dot(x, y)
	K.int_shape(xy)
		# (2, 4, 5)
	```

####	`batch_dot`

```python
def batch_dot(x, y, axes=None)
```

按批进行张量`x`和`y`的点积

-	参数
	-	`x`：按batch分块的数据
	-	`y`；同`x`
	-	`axes`：指定进行点乘的轴

```python
x_batch = K.ones(shape=(32, 20, 1))
y_batch = K.ones(shape=(32, 30, 20))
xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=[1, 2])
K.int_shape(xy_batch_dot)
	# (32, 1, 30)
```

####	`transpose`

```python
def transpose(x)
```

张量转置

####	`gather`

```python
def gather(reference, indices)
```

在给定的张量中检索给定下标的向量

-	参数
	-	`reference`：张量
	-	`indices`：整数张量，其元素为要查询的下标

-	返回值：同`reference`数据类型相同的张量

####	`max`

```python
def max(
	x,
	axis=None/int,
	keepdims=False
)
```

求张量中的最大值

####	`min`
```python
def min(x, axis=None, keepdims=False)
```

求张量中的最小值

####	`sum`

```python
sum(x, axis=None, keepdims=False)
```

计算张量中元素之和

####	`prod`

```python
prod(x, axis=None, keepdims=False)
```

计算张量中元素之积

####	`cumsum`

```python
def cumsum(x, axis=0)
```

在给定轴上求张量的累积和

####	`cumprod`

```python
cumprod(x, axis=0)
```

在给定轴上求张量的累积积

####	`var`

```python
def var(x, axis=None, keepdims=False)
```

在给定轴上计算张量方差

####	`std`

```python
def std(x, axis=None, keepdims=False)
```

在给定轴上求张量元素之标准差

####	`mean`

```python
def mean(x, axis=None, keepdims=False)
```

在给定轴上求张量元素之均值

####	`any`

```python
def any(x, axis=None, keepdims=False)
```

按位或，返回数据类型为uint8的张量（元素为0或1）

####	`all`

```python
def any(x, axis=None, keepdims=False)
```

按位与，返回类型为uint8de tensor

####	`argmax`

```python
def argmax(x, axis=-1)
```

在给定轴上求张量之最大元素下标

####	`argmin`

```python
def argmin(x, axis=-1)
```

在给定轴上求张量之最小元素下标

###	Element-Wise OPs

####	`square`

```python
def square(x)
```

逐元素平方

####	`abs`

```python
def abs(x)
```

逐元素绝对值

####	`sqrt`

```python
sqrt(x)
```

逐元素开方

####	`exp`

```python
def exp(x)
```

逐元素求自然指数

####	`log`

```python
def log(x)
```

逐元素求自然对数

####	`logsumexp`

```python
def logsumexp(x, axis=None, keepdims=False)
```

在给定轴上计算`log(sum(exp()))`

-	该函数在数值稳定性上超过手动计算`log(sum(exp()))`，可以
	避免由`exp`和`log`导致的上溢和下溢

####	`round`

```python
def round(x)
```

逐元素四舍五入

####	`sign`

```python
def sign(x)
```

逐元素求元素的符号

-	返回值
	-	`+1`
	-	`-1`

####	`pow`

```python
def pow(x, a)
```

逐元素求x的a次方

####	`clip`

```python
def clip(
	x,
	min_value,
	max_value
)
```

逐元素clip（将超出指定范围的数强制变为边界值）

####	`equal`

```python
def equal(x, y)
```

逐元素判相等关系

-	返回值：布尔张量OP

####	`not_equal`

```python
def not_equal(x, y)
```

逐元素判不等关系

####	`greater`

```python
def greater(x,y)
```

逐元素判断`x>y`关系

####	`greater_equal`

```python
def greater_equal(x,y)
```

逐元素判断`x>=y`关系

####	`lesser`

```python
def lesser(x,y)
```

逐元素判断`x<y`关系

####	`lesser_equal`

```python
def lesser_equal(x,y)
```

逐元素判断`x<=y`关系

####	`maximum`

```python
def maximum(x, y)
```

逐元素取两个张量的最大值

####	`minimum`

```python
def minimum(x, y)
```

逐元素取两个张量的最小值

####	`sin`

```python
def sin(x)
```

逐元素求正弦值

####	`cos`

```python
def cos(x)
```

逐元素求余弦值

###	变换OPs

####	`batch_normalization`

```python
def batch_normalization(
	x,
	mean,
	var,
	beta,
	gamma,
	epsilon=0.0001
)
```

对batch的数据进行*batch_normalization*，计算公式为：
$output = (x-mean)/(\sqrt(var)+\epsilon)*\gamma+\beta$

-	手动指定`mean`、`var`

####	`normalize_batch_in_training`

```python
def normalize_batch_in_training(
	x,
	gamma,
	beta,
	reduction_axes,
	epsilon=0.0001
)
```

对batch数据先计算其均值和方差，然后再进行
`batch_normalization`

####	`concatenate`

```python
def concatenate(tensors, axis=-1)
```

在给定轴上将一个列表中的张量串联为一个张量

####	`reshape`

```python
def reshape(x, shape)
```

将张量的shape变换为指定shape

####	`permute_dimensions`

```python
def permute_dimensions(
	x,
	pattern(tuple(int))
)
```

按照给定的模式重排一个张量的轴

-	参数
	-	`pattern`：代表维度下标的tuple如`(0, 2, 1)`

####	`resize_images`

```python
def resize_images(
	X,
	height_factor(uint),
	width_factor(uint),
	dim_ordering=None/'th'/'tf'
)
```

依据给定的缩放因子`height_factor`、`width_factor`，改变batch
数据（图片）的shape

-	参数
	-	`height_factor`/`width_factor`：正整数

####	`resize_volumes`

```python
def resize_volumes(
	X,
	depth_factor,
	height_factor,
	width_factor,
	dim_ordering
)
```

依据给定的缩放因子，改变一个5D张量数据的shape

####	`repeat_elements`

```python
def repeat_elements(x, rep, axis)
```

在给定轴上重复张量**元素**`rep`次

-	与`np.repeat`类似

####	`repeat`

```python
def repeat(x, n)
```

重复2D张量

####	`arange`

```python
def arange(
	start,
	stop=None,
	step=1,
	dtype="int32"
)
```

生成1D的整数序列张量

-	参数：同`np.arange`
	-	如果只有一个参数被提供了，则`0~stop`

-	返回值：默认数据类型是`int32`的张量

####	`tile`

```python
def tile(x, n)
```

将`x`在各个维度上重复`n[i]`次

####	`batch_flatten`

```python
def batch_flatten(x)
```

将n阶张量转变为2阶张量，第一维度保留不变

####	`expand_dims`

```python
def expand_dims(x, dim=-1)
```

在`dim`指定轴后增加一维（轴）

####	`squeeze`

```python
def squeeze(x, axis)
```

将`axis`指定的轴从张量中移除（保留轴上首组张量切片）

####	`temporal_padding`

```python
def temporal_padding(x, padding=1)
```

向3D张量中间那个维度的左右两端填充`padding`个0值

####	`asymmetric_temporal_padding`

```python
def asymmetric_temporal_padding(
	x,
	left_pad=1,
	right_pad=1
)
```

向3D张量中间的那个维度的左右分别填充0值

####	`spatial_2d_padding`

```python
def spatial_2d_padding(
	x,
	padding=(1, 1),
	dim_ordering='th'
)
```

向4D张量高度、宽度左右两端填充`padding[0]`和`padding[1]`
个0值


####	`spatial_3d_padding`

```python
def spatial_3d_padding(
	x,
	padding=(1, 1, 1),
	dim_ordering='th'
)
```

向5D张量深度、高度、宽度三个维度上填充0值

####	`stack`

```python
def stack(x, axis=0)
```

将列表`x`中张量堆积起来形成维度+1的新张量

####	`one_hot`

```python
def one_hot(indices, nb_classes)
```

为张量`indices`进行*one_hot*编码

-	参数
	-	`indices`：n维的整数张量
	-	`nb_classes`：*one_hot*编码列表
-	输出：n+1维整数张量，最后维为编码

####	`reverse`

```python
def reverse(x, axes)
```

将一个张量在给定轴上反转

####	`get_value`

```python
def get_value(x)
```

以NDA的形式返回张量的值

####	`batch_get_value`

```python
def batch_get_value(x)
```

以`[NDA]`的形式返回多个张量的值

####	`set_value`

```python
def set_value(x, value)
```

从NDA将值载入张量中

####	`batch_set_value`

```python
def batch_set_value(tuples)
```

将多个值载入多个张量变量中

####	`print_tensor`

```
def print_tensor(x, message='')
```

在求值时打印张量的信息，并返回原张量

####	`function`

```python
def function(inputs, outputs, updates=[])
```

实例化一个Keras函数

-	参数
	-	`inputs`：列表，其元素为占位符或张量变量
	-	`outputs`：输出张量的列表
	-	`updates`：张量列表

####	`gradients`

```python
def gradients(loss, variables)
```

返回`loss`函数关于`variables`的梯度

####	`stop_gradient`

```python
def stop_gradient(variables)
```

Returns `variables` but with zero gradient with respect to every other variables.

####	`rnn`

```python
def rnn(
	step_function,
	inputs,
	initial_states,
	go_backwards=False,
	mask=None,
	constants=None,
	unroll=False,
	input_length=None
)
```

在张量的时间维上迭代

-	参数：
	-	`inputs`：时域信号的张量，阶数至少为3

	-	`step_function`：每个时间步要执行的函数

		参数

		-	`input`：不含时间维张量，代表某时间步batch样本
		-	`states`：张量列表

		返回值
		
		-	`output`：形如```(samples, ...)```的张量
		-	`new_states`：张量列表，与`states`的长度相同	

	-	`initial_states`：包含`step_function`状态初始值

	-	`go_backwards`：逆向迭代序列

	-	`mask`：需要屏蔽的数据元素上值为1

	-	`constants`：按时间步传递给函数的常数列表

	-	`unroll`
		-	当使用TF时，RNN总是展开的'
		-	当使用Theano时，设置该值为`True`将展开递归网络

	-	`input_length`
		-	TF：不需要此值
		-	Theano：如果要展开递归网络，必须指定输入序列

-	返回值：形如`(last_output, outputs, new_states)`的张量
	-	`last_output`：RNN最后的输出
	-	`outputs`：每个在[s,t]点的输出对应于样本s在t时间的输出
	-	`new_states`: 每个样本的最后一个状态列表

####	`switch`

```python
def switch(
	condition,
	then_expression,
	else_expression
)
```

依据给定`condition`（整数或布尔值）在两个表达式之间切换


-	参数
	-	`condition`：标量张量
	-	`then_expression`：TensorFlow表达式
	-	`else_expression`: TensorFlow表达式

####	`in_train_phase`

```python
def in_train_phase(x, alt)
```

如果处于训练模式，则选择`x`，否则选择`alt`

-	注意`alt`应该与`x`的`shape`相同

####	`in_test_phase`

```python
def in_test_phase(x, alt)
```

如果处于测试模式，则选择`x`，否则选择`alt`

-	注意：`alt`应该与`x`的shape相同

##	预定义（激活）函数

####	`relu`

```python
def relu(
	x,
	alpha=0.0,
	max_value=None
)
```

修正线性单元

-	参数
	-	`alpha`：负半区斜率
	-	`max_value`: 饱和门限

####	`elu`

```python
def elu(x, alpha=1.0)
```

指数线性单元

-	参数
	-	`x`：输入张量
	-	`alpha`: 标量

####	`softmax`

```python
def softmax(x)
```

计算张量的softmax值

####	`softplus`

```python
def softplus(x)
```

返回张量的softplus值

####	`softsign`

```python
def softsign(x)
```

返回张量的softsign值

##	预定义目标函数

####	`categorical_crossentropy`

```python
def categorical_crossentropy(
	output,
	target,
	from_logits=False
)
```

计算`output`、`target`的Categorical Crossentropy（类别交叉熵）

-	参数
	-	`output`/`target`：shape相等

####	`sparse_categorical_crossentropy`

```python
def sparse_categorical_crossentropy(
	output,
	target,
	from_logits=False
)
```

计算`output`、`target`的Categorical Crossentropy（类别交叉熵）

-	参数
	-	`output`
	-	`target`：同`output` shape相等，需为整形张量


####	`binary_crossentropy`

```python
def binary_crossentropy(
	output,
	target,
	from_logits=False
)
```

计算输出张量和目标张量的交叉熵

####	`sigmoid`

```python
def sigmoid(x)
```

逐元素计算sigmoid值

####	`hard_sigmoid`

```python
def hard_sigmoid(x)
```

分段线性近似的sigmoid，计算速度更快

####	`tanh`

```python
def tanh(x)
```

逐元素计算tanh值 

####	`dropout`

```python
def dropout(x, level, seed=None)
```

随机将`x`中一定比例的值设置为0，并放缩整个Tensor

####	`l2_normalize`

```python
def l2_normalize(x, axis)
```

在给定轴上对张量进行L2范数规范化

####	`in_top_k`

```python
def in_top_k(predictions, targets, k)
```

判断目标是否在`predictions`的前k大值位置

参数

-	`predictions`：预测值张量
-	`targets`：真值张量
-	`k`：整数

####	`conv1d`

```python
def conv1d(
	x,
	kernel,
	strides=1,
	border_mode="valid"/"same",
	image_shape=None,
	filter_shape=None
)
```

1D卷积

-	参数
	-	`kernel`：卷积核张量
	-	`strides`：步长，整型
	-	`border_mode`
		-	"same"：
		-	"valid"：

####	`conv2d`

```python
def conv2d(
	x,
	kernel,
	strides=(1, 1),
	border_mode="valid"/"same",
	dim_ordering="th"/"tf",
	image_shape=None,
	filter_shape=None
)
```

2D卷积

-	参数
	-	`kernel`：卷积核张量
	-	`strides`：步长，长为2的tuple

####	`deconv2d`

```python
def deconv2d(
	x,
	kernel,
	output_shape,
	strides=(1, 1),
	border_mode="valid"/"same",
	dim_ordering="th"/"tf",
	image_shape=None,
	filter_shape=None
)
```

2D反卷积（转置卷积）

-	参数
	-	`x`：输入张量
	-	`kernel`：卷积核张量
	-	`output_shape`: 输出shape的1D的整数张量
	-	`strides`：步长，tuple类型

####	`conv3d`

```python
def conv3d(
	x,
	kernel,
	strides=(1, 1, 1),
	border_mode="valid"/"same",
	dim_ordering="th"/"tf",
	volume_shape=None,
	filter_shape=None
)
```

3D卷积

####	`pool2d`

```python
def pool2d(
	x,
	pool_size,
	strides=(1, 1),
	border_mode="valid"/"same",
	dim_ordering="th"/"tf",
	pool_mode="max"/"avg"
)
```

2D池化

-	参数
	-	`pool_size`：含有两个整数的tuple，池的大小
	-	`strides`：含有两个整数的tuple，步长
	-	`pool_mode`: “max”，“avg”之一，池化方式

####	`pool3d`

```python
def pool3d(
	x,
	pool_size,
	strides=(1, 1, 1),
	border_mode="valid"/"same",
	dim_ordering="th"/"tf",
	pool_mode="max"/"avg"
)
```

3D池化


####	`bias_add`

```python
def bias_add(x, bias, data_format=None)
```

为张量增加一个偏置项

####	`random_normal`

```python
def random_normal(
	shape,
	mean=0.0,
	stddev=1.0,
	dtype=None,
	seed=None
)
```

生成服从正态分布的张量

-	参数
	-	`mean`：均值
	-	`stddev`：标准差


####	`random_uniform`

```python
def random_uniform(
	shape,
	minval=0.0,
	maxval=1.0,
	dtype=None,
	seed=None
)
```

生成服从均匀分布值的张量

-	参数
	-	`minval`：上界
	-	`maxval`：上界

####	`random_binomial`

```python
def random_binomial(
	shape,
	p=0.0,
	dtype=None,
	seed=None
)
```

返回具有二项分布值的张量

-	参数
	-	`p`：二项分布参数

####	`truncated_normall`

```python
def truncated_normal(
	shape,
	mean=0.0,
	stddev=1.0,
	dtype=None,
	seed=None
)
```

生成服从截尾正态分布值的张量，即在距离均值两个标准差之外的
数据将会被截断并重新生成

####	`ctc_label_dense_to_sparse`

```python
def ctc_label_dense_to_sparse(labels, label_lengths)
```

将ctc标签从稠密形式转换为稀疏形式

####	`ctc_batch_cost`

```python
def ctc_batch_cost(
	y_true,
	y_pred,
	input_length,
	label_length
)
```

在batch上运行CTC损失算法

-	参数
	-	`y_true`：包含标签的真值张量
	-	`y_pred`：包含预测值或输出的softmax值的张量
	-	`input_length`：包含`y_pred`中每个batch的序列长
	-	`label_length`：包含`y_true`中每个batch的序列长张量

-	返回值：包含了每个元素的CTC损失的张量

####	`ctc_decode`

```python
def ctc_decode(
	y_pred,
	input_length,
	greedy=True,
	beam_width=None,
	dict_seq_lens=None,
	dict_values=None
)
```

使用贪婪算法或带约束的字典搜索算法解码softmax的输出

-	参数

	-	`y_pred`：包含预测值或输出的softmax值的张量
	-	`input_length`：包含`y_pred`中每个batch序列长的张量
	-	`greedy`：使用贪婪算法
	-	`dict_seq_lens`：`dic_values`列表中各元素的长度
	-	`dict_values`：列表的列表，代表字典

-	返回值：包含了路径可能性（以softmax概率的形式）张量

-	注意仍然需要一个用来取出argmax和处理空白标签的函数

####	`map_fn`

```python
def map_fn(fn, elems, name=None)
```

元素elems在函数fn上的映射，并返回结果

-	参数
	-	`fn`：函数
	-	`elems`：张量
	-	`name`：节点的名字

-	返回值：张量的第一维度等于`elems`，第二维度取决于`fn`

####	`foldl`

```python
def foldl(
	fn,
	elems,
	initializer=None,
	name=None
)
```

用fn从左到右连接它们，以减少`elems`值

-	参数

	-	`fn`：函数，例如：lambda acc, x: acc + x
	-	`elems`：张量
	-	`initializer`：初始化的值(elems[0])
	-	`name`：节点名

-	返回值：与`initializer`类型和形状一致

####	`foldr`

```python
def foldr(
	fn,
	elems,
	initializer=None,
	name=None
)
```

减少`elems`，用`fn`从右到左连接它们

-	参数

	-	`fn`：函数，例如：lambda acc, x: acc + x
	-	`elems`：张量
	-	`initializer`：初始化的值（elems[-1]）
	-	`name`：节点名

-	返回值：与`initializer`类型和形状一致

