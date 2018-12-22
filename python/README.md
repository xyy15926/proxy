#	Python笔记约定

##	函数书写声明

```python
return = func(essential(type), optional=defaults/type,
	*args, **kwargs)
```

###	格式说明

####	参数

-	`essential`参数：`essential(type)`
	-	没有`=`
	-	在参数列表开头
	-	`(type)`：表示参数可取值类型

-	`optional`参数：`optional=defaults/type`
	-	`defaults`若为具体值，表示默认参数值
		-	默认值首字母大写
	-	默认参数值为`None`
		-	函数内部有默认行为
		-	`None`（默认行为等价参数值）
	-	`type`：之后表示可能取值类型

-	`args`参数：`[参数名]=defaults/type`
	-	首参数值为具体值表示函数默认行为（不是默认值，`args`
		参数没有默认值一说）
	-	其后为可取参数值类型
	-	说明
		-	参数名仅是**标记**作用，不能使用关键字传参
		-	`[]`：代表参数“可选”

-	`kwargs`参数：`[param_name=defaults/type]`
	-	参数默认为可选参数，格式规则类似`args`参数

-	POSITION_ONLY参数：`[param_name](defaults/type)`
	-	POSITION_ONLY参数同样没有默认值一说，只是表示默认
		行为方式（对应参数值）

补充：
-	参数名后有`?`表示参数名待确定
-	参数首字母大写表示唯一参数

####	返回值

返回值类型由返回对象**名称**蕴含

####	对象类型

-		`obj(type)`：`type`表示**包含**元素类型

####	其他

-	DF对象和Series对象都具有的函数属性列出DF对象

##	常用参数说明

以下常用参数如不特殊注明，按照此解释

###	Pandas

-	`axis=0/1/"index"/"columns"`
	-	含义：作用方向（轴）
	-	默认：`0/"index"`，一般表示row-wise（行变动）方向

-	`inplace=False/True`
	-	含义：是否直接在原对象更改
	-	默认：`False`，不更改，返回新DF对象（为`True`时无返回值）
	-	其他
		-	大部分df1.func()类型函数都有这个参数

-	`level=0/1/level_name...`
	-	含义：用索引层级
	-	默认：部分默认为`0`（顶层级）（也有默认为底层级），
		所以有时会如下给出默认值
		-	`t`（top）：顶层级`0`（仅表意）
		-	`b`（bottom）：底层级`-1`（仅表意）
		-	默认值为`None`表示所有层级

###	Matplotlib

#	todo

-	`data=dict/pd.DataFrame`

	-	其他
		-	属于kwargs中参数
		-	传参时，相应键值对替代对应参数

###	Numpy

-	`size=None(1)/int/tuple(int)`

	-	含义：ndarray形状
	-	默认：一般`None`，返回一个值

-	`dtype=None/str/np.int/np.float...`

	-	含义：ndarray中数据类型
	-	默认值：`None`，有内部操作，选择合适、不影响精度类型
	-	其他
		-	可以是字符串形式，也可以是`np.`对象形式
-	`order = "C"/"F"/"K"/"A"`
	-	含义：NDA对象在内存中的存储方式
		-	"C"：`C`存储方式，行优先
		-	"F"：`Fortran`存储方式，列优先
		-	"K"：原为"C"/"F"方式则保持不变，否则按照较接近
			方式
		-	"A"：除非原为"F"方式，否则为"C"方式
	-	默认值："C"/"K"

>	Numpy包中大部分应该是调用底层包，参数形式不好确认

###	threading

-	`block/blocking = True/False`

	-	含义：是否阻塞
	-	默认：大部分为`True`（阻塞）

-	`timeout = None/num`

	-	含义：延迟时间，单位一般是秒
	-	默认：None，无限时间

###	Tensorflow

-	`name = None/str`
	-	含义：Operations名
	-	默认：有的为None，有的为Operation类型，但效果一样，
		没有传参时使用Operation类型（加顺序后缀）

-	`axis = None/0/int`
	-	含义：指定张量轴
	-	默认
		-	`None`：大部分，表示在整个张量上运算
		-	`0`：有些运算难以推广到整个张量，表示在首轴（维）

-	`keepdims=False/True`
	-	含义：是否保持维度数目
	-	默认：`False`不保持

###	Keras

-	`seed=None/int`
	-	含义：随机数种子

-	`padding="valid"/"same"/"causal"`
	-	含义：补0策略
		-	"valid"：只进行有效有效卷积，忽略边缘数据，输入
			数据比输出数据shape减小
		-	"same"：保留边界处卷积结果，输入数据和数据shape
			相同
		-	"causal"：产生膨胀（因果卷积），即`output[t]`
			不依赖`input[t+1:]`，对不能违反时间顺序的时序
			信号建模时有用
	-	默认："valid"

####	Layers

-	`input_shape=None/(int,...)`
	-	含义：输入数据shape
		-	Layers只有首层需要传递该参数，之后层可自行推断
		-	传递tuple中`None`表示改维度边长
	-	默认：`None`，由Layers自行推断

-	`data_format=None/"channels_last"/"channels_first"`
	-	含义：通道轴位置
		-	类似于`dim_ordering`，但是是Layer参数
	-	默认
		-	大部分：`None`由配置文件（默认"channels_last"）
		、环境变量决定
		-	Conv1DXX："channels_last"
		-	其实也不一定，最好每次手动指定。。。

-	`dim_ordering=None/"th"/"tf"`
	-	含义：中指定channals轴位置(`th`batch后首、`tf`尾）
	-	默认：`None`以Keras配置为准
	-	注意：Deprecated，Keras1.x中使用


#####	Conv Layers

-	`filters(int)`
	-	含义：输出维度
		-	对于卷积层，就是卷积核数目，因为卷积共享卷积核
		-	对于局部连接层，是卷积核**组数**，不共享卷积核
			，实际上对每组有很多不同权重

-	`kernel_size(int/(int)/[int])`
	-	含义：卷积核形状，单值则各方向等长

-	`strides(int/(int)/[int])`
	-	含义：卷积步长，单值则各方向相同
	-	默认：`1`移动一个步长

-	`dilation_rate(int/(int)/[int])`
	-	含义：膨胀比例
		-	即核元素之间距离
		-	`dilation_rate`、`strides`最多只能有一者为1，
			即核膨胀、移动扩张最多只能出现一种
	-	默认：`1`不膨胀，核中个元素相距1

-	`use_bias=True/False`
	-	含义：是否使用偏置项
	-	默认：`True`使用偏置项

-	`activation=str/func`
	-	含义：该层激活函数
		-	`str`：预定义激活函数字符串
		-	`func`：自定义element-wise激活函数
	-	默认：`None`不做处理（即线性激活函数）

-	`kernel_initializer=str/func`
	-	含义：权值初始化方法
		-	`str`：预定义初始化方法名字符串
			（参考Keras Initializer）
		-	`func`：初始化权重的初始化器
	-	默认：`glorot_uniform`初始化为平均值

-	`bias_initializer=str/func`
	-	含义：偏置初始化方法
		-	`str`：预定义初始化方法名字符串
		-	`func`：初始化权重的初始化器
	-	默认：`zeros`初始化为全0

-	`kernel_regularizer=None/obj`
	-	含义：施加在权重上的正则项
		（参考Keras Regularizer对象）
	-	默认：`None`不使用正则化项

-	`bias_regularizer=None/obj`
	-	含义：施加在偏置上的正则项
		（参考Keras Regularizer对象）
	-	默认：`None`不使用正则化项

-	`activity_regularizer=None/obj`
	-	含义：施加在输出上的正则项
		（参考Keras Regularizer对象）
	-	默认：`None`不使用正则化项

-	`kernel_constraint=None/obj`
	-	含义：施加在权重上的约束项
		（参考Keras Constraints）
	-	默认：`None`不使用约束项

-	`bias_constraint=None`
	-	含义：施加在偏置上的约束项
		（参考Keras Constraints）
	-	默认：`None`不使用约束项

