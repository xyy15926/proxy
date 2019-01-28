#	RNN

## RNN

```python
keras.layers.RNN(
	cell,
	return_sequences=False,
	return_state=False,
	go_backwards=False,
	stateful=False,
	unroll=False
)
```

循环神经网络层基类：抽象类、无法实例化对象

-	参数

	-	 `cell`：RNN单元实例、列表，为RNN单元列表时，单元
			堆叠放置，实现高效堆叠RNN

	-	`return_sequences`：返回输出序列最后值/全部序列

	-	`return_state`：是否返回最后一个状态

	-	`go_backwards`：是否向后处理输入序列并返回相反的序列。

	-	`stateful`：批次中索引`i`处的每个样品的最后状态将
		用作下一批次中索引`i`样品的初始状态

	-	`unroll`：是否将网络展开（否则将使用符号循环）
		-	展开可以加速 RNN，但它往往会占用更多的内存
		-	展开只适用于短序列

	-	`input_dim`：输入的维度（整数）
		-	将此层用作模型中的第一层时，此参数是必需的
			（或者关键字参数 `input_shape`）

	-	`input_length`： 输入序列的长度，在恒定时指定
		-	如果你要在上游连接 `Flatten` 和 `Dense` 层，则
			需要此参数（没有它，无法计算全连接输出的尺寸）
		-	如果循环神经网络层不是模型中的第一层，则需要在
			第一层的层级指定输入长度
			（或通过关键字参数`input_shape`）

-	输入：`(batch_size, timesteps, input_dim)`3D张量

-	输出
	-	`return_state=True`：则返回张量列表，第一个张量为
		输出、剩余的张量为最后的状态，每个张量的尺寸为
		`(batch_size, units)`。
	-	`return_state=False`：`(batch_size, units)`2D张量

说明

-	屏蔽覆盖：支持以可变数量的时间步长对输入数据进行屏蔽覆盖

-	使用状态：可以将 RNN 层设置为 `stateful`（有状态的）
	-	这意味着针对一批中的样本计算的状态将被重新用作下一批
		样品的初始状态
	-	这假定在不同连续批次的样品之间有一对一的映射。

	-	为了使状态有效：
		- 在层构造器中指定 `stateful=True`。
		- 为模型指定一个固定的批次大小
			-	顺序模型：为模型的第一层传递一个
				`batch_input_shape=(...)` 参数
			-	带有Input层的函数式模型，为的模型的**所有**i
				第一层传递一个`batch_shape=(...)`，这是输入
				预期尺寸，包括批量维度
		-	在调用 `fit()` 是指定 `shuffle=False`。

	-	要重置模型的状态，请在特定图层或整个模型上调用
		`.reset_states()`

-	初始状态

	-	通过使用关键字参数`initial_state`调用它们来符号化地
		指定 RNN 层的初始状态（值应该是表示RNN层初始状态的
		张量或张量列表）
	-	通过调用带有关键字参数`states`的`reset_states`方法
		来数字化地指定 RNN 层的初始状态（值应该是一个代表RNN
		层初始状态的NDA/[NDA]）

-	RNN单元对象需要具有

	-	`call(input_at_t, states_at_t)`方法，它返回
		`(output_at_t, states_at_t_plus_1)`，单元的调用
		方法也可以采用可选参数 `constants`
	-	 `state_size`属性
		-	单个整数（单个状态）：在这种情况下，它是循环层
			状态大小（应该与单元输出的大小相同）
		-	整数的列表/元组（每个状态一个大小）：第一项应该
			与单元输出的大小相同

-	传递外部常量

	-	使用`RNN.call`以及`RNN.call`的`constants`关键字
		参数将*外部*常量传递给单元
	-	要求`cell.call`方法接受相同的关键字参数`constants`，
		这些常数可用于调节附加静态输入（不随时间变化）上的
		单元转换，也可用于注意力机制

例子

```python
class MinimalRNNCell(keras.layers.Layer):
	# 定义RNN细胞单元（网络层子类）
	def init(self, units, **kwargs):
		self.units = units
		self.state_size = units
		super(MinimalRNNCell, self).init(**kwargs)

	def build(self, input_shape):
		self.kernel = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer="uniform",
			name="kernel"
		)
		self.recurrent_kernel = self.add_weight(
			shape=(self.units, self.units),
			initializer="uniform",
			name="recurrent_kernel")
		self.built = True

	def call(self, inputs, states):
		prev_output = states[0]
		h = K.dot(inputs, self.kernel)
		output = h + K.dot(prev_output, self.recurrent_kernel)
		return output, [output]

cell = MinimalRNNCell(32)
	# 在RNN层使用这个单元：
x = keras.Input((None, 5))
layer = RNN(cell)
y = layer(x)


cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
	# 用单元格构建堆叠的RNN的方法：
x = keras.Input((None, 5))
layer = RNN(cells)
y = layer(x)
```

##	SimpleRNN

```python
keras.layers.SimpleRNN(
	units,
	activation="tanh",
	use_bias=True,
	kernel_initializer="glorot_uniform",
	recurrent_initializer="orthogonal",
	bias_initializer="zeros",
	kernel_regularizer=None,
	recurrent_regularizer=None,
	bias_regularizer=None,
	activity_regularizer=None,
	kernel_constraint=None,
	recurrent_constraint=None,
	bias_constraint=None,
	dropout=0.0,
	recurrent_dropout=0.0,
	return_sequences=False,
	return_state=False,
	go_backwards=False,
	stateful=False,
	unroll=False
)
```

完全连接的RNN，其输出将被反馈到输入。

-	参数

	-	`units`：正整数，输出空间的维度。

	-	`activation`：要使用的激活函数
		-	`tanh`：默认
		-	`None`：则不使用激活函数，即线性激活：`a(x) = x`

	-	`use_bias`：布尔值，该层是否使用偏置向量。

	-	`kernel_initializer`：*kernel*权值矩阵的初始化器

	-	`recurrent_initializer`：*recurrent_kernel*权值矩阵

	-	`bias_initializer`：偏置向量的初始化器

	-	`kernel_regularizer`：运用到*kernel*权值矩阵的正则化
		函数

	-	`recurrent_regularizer`：运用到 `recurrent_kernel` 权值
		矩阵的正则化函数

	-	`bias_regularizer`：运用到偏置向量的正则化函数

	-	`activity_regularizer`：运用到层输出（它的激活值）的
		正则化函数

	-	`kernel_constraint`：运用到*kernel*权值矩阵的约束函数

	-	`recurrent_constraint`：运用到*recurrent_kernel*权值矩阵
		的约束函数

	-	`bias_constraint`： 运用到偏置向量的约束函数

	-	`dropout`：单元的丢弃比例，用于输入的线性转换
		-	在*0-1*之间的浮点数

	-	`recurrent_dropout`：单元的丢弃比例，用于循环层状态线性
		转换

	-	`return_sequences`：返回输出序列中的全部序列
		-	默认：返回最后最后一个输出

	-	`return_state`：除输出之外是否返回最后一个状态

	-	`go_backwards`：向后处理输入序列并返回相反的序列

	-	`stateful`：批次中索引 i 处的每个样品的最后状态，将用作
		下一批次中索引 i 样品的初始状态

	-	`unroll`：展开网络

##	GRU

```python
keras.layers.GRU(
	units,
	activation="tanh",
	recurrent_activation="hard_sigmoid",
	use_bias=True,
	kernel_initializer="glorot_uniform",
	recurrent_initializer="orthogonal",
	bias_initializer="zeros",
	kernel_regularizer=None,
	recurrent_regularizer=None,
	bias_regularizer=None,
	activity_regularizer=None,
	kernel_constraint=None,
	recurrent_constraint=None,
	bias_constraint=None,
	dropout=0.0,
	recurrent_dropout=0.0,
	implementation=1,
	return_sequences=False,
	return_state=False,
	go_backwards=False,
	stateful=False,
	unroll=False,
	reset_after=False
)
```

门限循环单元网络

-	说明：有两种变体

	-	默认的基于*1406.1078v3*，并且在矩阵乘法之前将复位门
		应用于隐藏状态

	-	另一种基于*1406.1078v1*并且顺序倒置
	
		-	兼容*CuDNNGRU(GPU-only)*，并且允许在 CPU 上进行
			推理

		-	对于*kernel*和*recurrent_kernel*有可分离偏置
			`reset_after=True`和`recurrent_activation=sigmoid` 。

-	参数

	-	`recurrent_activation`：用于循环时间步的激活函数

	-	`implementation`：实现模式

		-	`1`：将把它的操作结构化为更多的小的点积和加法操作

		-	`2`：将把它们分批到更少，更大的操作中

		-	这些模式在不同的硬件和不同的应用中具有不同的性能配置
			文件

	-	`reset_after`：GRU公约，在矩阵乘法之后使用重置门

-	参考文献

	-	[Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
	-	[On the Properties of Neural Machine Translation：Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259)
	-	[Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/abs/1412.3555v1)
	-	[A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)

##	LSTM

```python
keras.layers.LSTM(
	units,
	activation="tanh",
	recurrent_activation="hard_sigmoid",
	use_bias=True,
	kernel_initializer="glorot_uniform",
	recurrent_initializer="orthogonal",
	bias_initializer="zeros",
	unit_forget_bias=True,
	kernel_regularizer=None,
	recurrent_regularizer=None,
	bias_regularizer=None,
	activity_regularizer=None,
	kernel_constraint=None,
	recurrent_constraint=None,
	bias_constraint=None,
	dropout=0.0,
	recurrent_dropout=0.0,
	implementation=1,
	return_sequences=False,
	return_state=False,
	go_backwards=False,
	stateful=False,
	unroll=False
)
```

长短期记忆网络层（Hochreiter 1997）

-	参数

	-	`unit_forget_bias`
		-	`True`：初始化时，将忘记门的偏置加 1，同时还会强制
			`bias_initializer="zeros"`（这个建议来自
			[Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)）

-	参考文献

	-	[Long short-term memory](http://www.bioinf.jku.at/publications/older/2604.pdf) (original 1997 paper)
	-	[Learning to forget：Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
	-	[Supervised sequence labeling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
	-	[A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)

##	ConvLSTM2D

```python
keras.layers.ConvLSTM2D(
	filters(int),
	kernel_size(tuple(int, int)),
	strides=(1, 1),
	padding='valid',
	data_format=None,
	dilation_rate=(1, 1),
	activation='tanh',
	recurrent_activation='hard_sigmoid',
	use_bias=True,
	kernel_initializer='glorot_uniform',
	recurrent_initializer='orthogonal',
	bias_initializer='zeros',
	unit_forget_bias=True,
	kernel_regularizer=None,
	recurrent_regularizer=None,
	bias_regularizer=None,
	activity_regularizer=None,
	kernel_constraint=None,
	recurrent_constraint=None,
	bias_constraint=None,
	return_sequences=False,
	go_backwards=False,
	stateful=False,
	dropout=0.0,
	recurrent_dropout=0.0
)
```

卷积LSTM：它类似于LSTM层，但输入变换和循环变换都是卷积的

-	说明

	-	当前的实现不包括单元输出的反馈回路

-	参数

	-	`dilation_rate`：用于膨胀卷积的膨胀率
		-	`stride!=1`与`dilation_rate!=1`两者不兼容。

-	输入尺寸

	-	`data_format="channels_first"`：尺寸为
		`(batch, time, channels, rows, cols)`
	-	`data_format="channels_last"`：尺寸为
		`(batch, time, rows, cols, channels)`

-	输出尺寸

	-	`return_sequences=True`

		-	`data_format="channels_first"`：返回尺寸为
			`(batch, time, filters, output_row, output_col)`

		-	`data_format="channels_last"`：返回尺寸为
			`(batch, time, output_row, output_col, filters)`

	-	`return_seqences=False`

		-	`data_format ="channels_first"`：返回尺寸为
			`(batch, filters, output_row, output_col)`

		-	`data_format="channels_last"`：返回尺寸为
			`(batch, output_row, output_col, filters)`

-	参考文献

	-	[Convolutional LSTM Network：A Machine Learning Approach for Precipitation Nowcasting](http://arxiv.org/abs/1506.04214v1)

##	SimpleRNNCell

```python
keras.layers.SimpleRNNCell(
	units,
	activation='tanh',
	use_bias=True,
	kernel_initializer='glorot_uniform',
	recurrent_initializer='orthogonal',
	bias_initializer='zeros',
	kernel_regularizer=None,
	recurrent_regularizer=None,
	bias_regularizer=None,
	kernel_constraint=None,
	recurrent_constraint=None,
	bias_constraint=None,
	dropout=0.0,
	recurrent_dropout=0.0
)
```

*SimpleRNN*的单元类

##	GRUCell

```python
keras.layers.GRUCell(
	units,
	activation='tanh',
	recurrent_activation='hard_sigmoid',
	use_bias=True,
	kernel_initializer='glorot_uniform',
	recurrent_initializer='orthogonal',
	bias_initializer='zeros',
	kernel_regularizer=None,
	recurrent_regularizer=None,
	bias_regularizer=None,
	kernel_constraint=None,
	recurrent_constraint=None,
	bias_constraint=None,
	dropout=0.0,
	recurrent_dropout=0.0,
	implementation=1
)
```

*GRU*层的单元类

##	LSTMCell

```python
keras.layers.LSTMCell(
	units,
	activation='tanh',
	recurrent_activation='hard_sigmoid',
	use_bias=True,
	kernel_initializer='glorot_uniform',
	recurrent_initializer='orthogonal',
	bias_initializer='zeros',
	unit_forget_bias=True,
	kernel_regularizer=None,
	recurrent_regularizer=None,
	bias_regularizer=None,
	kernel_constraint=None,
	recurrent_constraint=None,
	bias_constraint=None,
	dropout=0.0,
	recurrent_dropout=0.0,
	implementation=1
)
```

LSTM层的单元类

##	StackedRNNCells

```python
keras.layers.StackedRNNCells(cells)
```

将一堆RNN单元表现为一个单元的封装器

-	说明

	-	用于实现高效堆叠的 RNN。

-	参数

	-	`cells`：RNN 单元实例的列表

例子


```python
cells = [
    keras.layers.LSTMCell(output_dim),
    keras.layers.LSTMCell(output_dim),
    keras.layers.LSTMCell(output_dim),
]

inputs = keras.Input((timesteps, input_dim))
x = keras.layers.RNN(cells)(inputs)
```
##	CuDNNGRU

```python
keras.layers.CuDNNGRU(
	units,
	kernel_initializer='glorot_uniform',
	recurrent_initializer='orthogonal',
	bias_initializer='zeros',
	kernel_regularizer=None,
	recurrent_regularizer=None,
	bias_regularizer=None,
	activity_regularizer=None,
	kernel_constraint=None,
	recurrent_constraint=None,
	bias_constraint=None,
	return_sequences=False,
	return_state=False,
	stateful=False
)
```

由 [CuDNN](https://developer.nvidia.com/cudnn) 
支持的快速*GRU*实现

-	说明

	-	只能以*TensorFlow*后端运行在*GPU*上

##	CuDNNLSTM

```python
keras.layers.CuDNNLSTM(
	units,
	kernel_initializer='glorot_uniform',
	recurrent_initializer='orthogonal',
	bias_initializer='zeros',
	unit_forget_bias=True,
	kernel_regularizer=None,
	recurrent_regularizer=None,
	bias_regularizer=None,
	activity_regularizer=None,
	kernel_constraint=None,
	recurrent_constraint=None,
	bias_constraint=None,
	return_sequences=False,
	return_state=False,
	stateful=False
)
```

由 [CuDNN](https://developer.nvidia.com/cudnn)
支持的快速*LSTM*实现

-	说明

	-	只能以*TensorFlow*后端运行在*GPU*上

