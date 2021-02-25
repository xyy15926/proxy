---
title: Universal Functions
categories:
  - Python
  - Numpy
  - Ufunc
tags:
  - Python
  - Numpy
  - NDArray
  - Data Science
date: 2021-02-20 08:40:31
updated: 2021-02-20 08:40:31
toc: true
mathjax: true
description: 
---

##	Universal Functions

-	*UFunc*：在数组上执行逐元素运算函数
	-	支持广播、类型映射等
	-	可视为是函数的向量化包装
	-	基本*ufunc*在标量上执行操作，更泛化的*ufunc*也可以
		在以子数组为基本元素进行操作

-	numpy中的*ufunc*是`np.ufunc`的实例
	-	许多内建的*ufunc*是通过C编译实现的
	-	可以通过`np.frompyfunc`工厂方法自定义*ufunc*实例

###	说明

-	Broadcasting：4条广播规则用于处理不同shape的数组
	-	非维数最大者在`shape`前用`1`补足
	-	输出的`shape`中各维度是各输入对应维度最大值
	-	各输入的维度同输出对应维度相同、或为`1`
	-	输入中维度为`1`者，对应的（首个）数据被用于沿该轴的
		所有计算
		（即对应的`stride`为`0`，*ufunc*不step along该维度）

-	Internal Buffers
	-	用于数据非对齐、数据交换、数据类型转换场合
	-	`.setbufsize(size)`：基于线程设置内部缓冲，缺省为
		`10,000`元素

-	错误处理
	-	设置硬件平台上注册的错误处理，如：除零错误
	-	基于线程设置

	|Function|Desc|
	|-----|-----|
	|`np.seterr([all,divide,over,under,invalid])`|设置浮点错误处理|
	|`np.seterrcall(func)`|设置浮点错误回调或log|

-	类型转换规则
	-	各*ufunc*内部维护列表，给出适用的输入类型（组合）、
		相应的输出类型
		（可通过`.types`属性查看）
	-	当*ufunc*内部列表中没有给定的输入类型组合时，则需要
		进行safely类型转换

-	可以通过`np.can_cast`函数确定是否进行类型转换
	-	`"S", "U", "V"`类型不能支持*ufunc*运算
	-	标量-数组操作使用不同类型转换规则确保标量不会降低
		数组精度，除非标量和数组属于同一类型体系

	```python
	def print_casting(ntypes):
		print("X")
		for char in ntypes:
			print(char, end=" ")
		print("")
		for row in ntypes:
			print(row, end=" ")
			for col in ntypes:
				print(int(np.can_cast(row, col)), end=" ")
			print("")
	print_casting(np.typecodes["All"])
	```

###	UFunc 说明

-	*core dimension*：核心维度，*ufunc*执行操作所在的维度
	-	核心维度一般使用元组表示
		-	对一般*ufunc*：核心维度为空元组
		-	对广义*ufunc*：核心维度为非空元组、空元组
	-	*signature*：签名，包含*ufunc*涉及的输出操作数和输出
		操作数的核心维度字符串，如：`(i,),(j,)->()`
	-	签名中各输入操作数的对应核心维度大小必须相同，移除后
		剩余的循环维度共同广播，加上输出操作数的核心维度得到
		输出结果shape

-	*loop dimension*：循环维度，除核心维度之外的维度

> - 这些术语来自*Perl Vector Library*
> - <https://numpy.org/doc/1.17/reference/c-api.generalized-ufuncs.html>

####	UFunc可选参数

```python
NDA = def numpy.<ufunc>(
	x1 [,x2], /,
	[out1, out2,], out, *,
	where=True,
	casting="same_kind",
	order="K",
	dtype=None,
	subok=True,
	[signature, extobj]
)
```

-	`where=True/False/Array[bool]`
	-	此参数不用于对子数组做操作的广义*ufunc*

-	`keepdims=False/True`
	-	对广义*ufunc*，只在输入操作数上有相同数量核心维度、
		输出操作数没有核心维度（即返回标量）时使用

-	`axes=tuple/int`
	-	含义：广义*ufunc*执行操作、存储结果所在的轴序号
		-	`[tuple]`：各元组为各输入操作数应被执行操作、
			输出操作数存储结果的轴的序号
		-	`[int]`：广义*ufunc*在1维向量上执行操作时，可以
			直接使用整形
	-	若广义*ufunc*的输出操作数均为标量，可省略其对应元组

-	`axis=int`
	-	含义：广义*ufunc*执行操作所在的single轴序号
		-	`int`：广义*ufunc*在相同的轴`axis`上执行操作，
			等价于`axes=[(axis,),(axis,),...]`

-	`signature=np.dtype/tuple[np.dtype]/str`
	-	含义：指示*ufunc*的输入、输出的数据类型，
	-	对于底层计算1维loop，是通过比较输入的数据类型，找到
		让所有输入都能安全转换的数据类型
		-	此参数允许绕过查找，直接指定loop
	-	可通过`ufunc.types`属性查看可用的signature列表

-	`extobj=list`
	-	含义：指定*ufunc*的缓冲大小、错误模式整数、错误处理
		回调函数
		-	`list`：长度为1、或2、或3的列表
	-	默认这些值会在对应线程字典中查找，此参数可以通过更
		底层的控制
		-	可优化在小数组上大量*ufunc*的调用

> - 部分参数含义通用，参见*README*

###	UFunc属性

|Attr|Desc|
|-----|-----|
|`ufunc.nin`|输入数量|
|`ufunc.nout`|输出数量|
|`ufunc.nargs`|参数数量|
|`ufunc.ntypes`|类型数量|
|`ufunc.types`|*input->output*列表|
|`ufunc.identity`|标志值|
|`ufunc.signature`|广义*ufunc*执行操作所在的核心元素的定义|

###	UFunc方法

|Method|Desc|
|-----|-----|
|`ufunc.reduce(a[,axis,dtype,out,...])`|通过沿轴应用*ufunc*缩减维度|
|`ufunc.accumulate(array[,axis,dtype,out])`|累加所有元素的计算结果|
|`ufunc.reduceat(a,indice[,axis,dtype,out])`|在single轴指定切片上执行reduce|
|`ufunc.outer(A,B,**kwargs)`|在分属`A,B`的元素对上应用*ufunc*|
|`ufunc.at(a,indices[,b])`|在`indices`处在位无缓冲执行操作|

-	所有*ufunc*都有4个方法，但是这些方法只在标量*ufunc*、
	包含2输入参数、1输出参数里有价值，否则导致`ValueError`

###	预定义UFunc

-	numpy中包含超过60种*ufunc*
	-	部分*ufunc*在相关运算标记调用时，会被自动调用

####	Math操作

|Function|Desc|
|-----|-----|
|`add(x1,x2,/[out,where,casting,order,...])`||
|`subtract(x1,x2,/[,out,where,casting,...])`||
|`multiply(x1,x2,/[,out,where,casting,...])`||
|`divide(x1,x2,/[,out,where,casting,...])`||
|`true_devide(x1,x2,/[,out,where,casting,...])`||
|`floor_devide(x1,x2,/[,out,where,casting,...])`||
|`logaddexp(x1,x2,/[,out,where,casting,...])`|`ln(x1+x2)`|
|`logaddexp2(x1,x2,/[,out,where,casting,...])`|`log_2 (x1+x2)`|
|`negative(x,/[,out,where,casting,order,...])`||
|`positive(x,/[,out,where,casting,order,...])`||
|`power(x1,x2,/[,out,where,casting,order,...])`|`x1^x2`|
|`flaot_power(x1,x2,/[,out,where,casting,...])`|`x1^x2`|
|`remainder(x1,x2,/[,out,where,casting,...])`|求余/取模|
|`mod(x1,x2,/[,out,where,casting,order,...])`|求余/取模|
|`fmod(x1,x2,/[,out,where,casting,order,...])`|求余/取模|
|`divmod(x1,x2,/[,out1,out2],/[out,...])`||
|`absolute(x,/[,out,where,casting,order,...])`||
|`fabs(x,/[,out,where,casting,order,...])`||
|`rint(x,/[,out,where,casting,order,...])`||
|`sign(x,/[,out,where,casting,order,...])`||
|`heaviside(x1,x2,/[,out,where,casting,...])`|阶跃函数|
|`conj(x,/[,out,where,casting,...])`|对偶|
|`exp(x,/[,out,where,casting,order,...])`||
|`exp2(x,/[,out,where,casting,order,...])`||
|`log(x,/[,out,where,casting,order,...])`||
|`log2(x,/[,out,where,casting,order,...])`||
|`log10(x,/[,out,where,casting,order,...])`||
|`expm1(x,/[,out,where,casting,order,...])`|计算`exp(x)-1`|
|`log1p(x,/[,out,where,casting,order,...])`|计算`ln(x+1)`|
|`sqrt(x,/[,out,where,casting,order,...])`|非负平方根|
|`square(x,/[,out,where,casting,order,...])`||
|`cbrt(x,/[,out,where,casting,order,...])`|立方根|
|`reciprocal(x,/[,out,where,casting,order,...])`|倒数|
|`gcd(x,/[,out,where,casting,order,...])`|最大公约数|
|`lcm(x,/[,out,where,casting,order,...])`|最小公倍数|

-	`out`参数可用于节省内存，如：`G=A*B+C`
	-	等价于：`t1=A*B; G=t1+C; del t1;`
	-	可利用`out`节省中间过程内存：`G=A*B; np.add(G,C,G)`

####	三角函数

|Function|Desc|
|-----|-----|
|`sin(x,/[,out,where,casting,order,...])`||
|`cos(x,/[,out,where,casting,order,...])`||
|`tan(x,/[,out,where,casting,order,...])`||
|`arcsin(x,/[,out,where,casting,order,...])`||
|`arccos(x,/[,out,where,casting,order,...])`||
|`arctan(x,/[,out,where,casting,order,...])`||
|`arctan2(x1,x2,/[,out,where,casting,order,...])`|考虑象限下，`arctan(x1/x2)`|
|`hypot(x1,x2,/[,out,where,casting,order,...])`|计算斜边|
|`sinh(x,/[,out,where,casting,order,...])`|双曲正弦|
|`cosh(x,/[,out,where,casting,order,...])`||
|`tanh(x,/[,out,where,casting,order,...])`||
|`arcsinh(x,/[,out,where,casting,order,...])`||
|`arccosh(x,/[,out,where,casting,order,...])`||
|`arctanh(x,/[,out,where,casting,order,...])`||
|`deg2rad(x,/[,out,where,casting,order,...])`|角度转换为弧度|
|`rad2deg(x,/[,out,where,casting,order,...])`|弧度转换为角度|

####	Bit-twiddling函数

|Function|Desc|
|-----|-----|
|`bitwise_and(x1,x2,/[,out,where,...])`||
|`bitwise_or(x1,x2,/[,out,where,...])`||
|`bitwise_xor(x1,x2,/[,out,where,...])`||
|`invert(x,/[,out,where,casting,...])`||
|`left_shift(x1,x2,/[,out,where,casting...])`||
|`left_shift(x1,x2,/[,out,where,casting...])`||

####	比较函数

|Function|Desc|
|-----|-----|
|`greater(x1,x2,/[,out,where,casting,...])`||
|`greater_equal(x1,x2,/[,out,where,casting,...])`||
|`less(x1,x2,/[,out,where,casting,...])`||
|`less_equal(x1,x2,/[,out,where,casting,...])`||
|`not_equal(x1,x2,/[,out,where,casting,...])`||
|`equal(x1,x2,/[,out,where,casting,...])`||
|`logical_and(x1,x2,/[,out,where,casting,...])`|逐元素`and`|
|`logical_or(x1,x2,/[,out,where,casting,...])`||
|`logical_xor(x1,x2,/[,out,where,casting,...])`||
|`logical_not(x1,x2,/[,out,where,casting,...])`||
|`maximum(x1,x2,/[,out,where,casting,...])`|逐元素选择较大者|
|`minimum(x1,x2,/[,out,where,casting,...])`|逐元素选择较小者|
|`fmax(x1,x2,/[,out,where,casting,...])`|逐元素选择较大者|
|`fmin(x1,x2,/[,out,where,casting,...])`|逐元素选择较小者|


-	数值比较
	-	`np.equal()`更多应用于整形比较，比较浮点使用
		`np.allclose()`更合适

-	逻辑运算符
	-	`&`、`|`、`~`：逐元素逻辑运算
		-	优先级高于比较运算符
	-	`and`、`or`、`not`：整个数组的逻辑运算

-	`np.maximum()`、`np.minimum()`函数
	-	`max()`寻找最大值效率比`np.maximum.reduce()`低，同样
		`min()`效率更低

####	Floating函数

|Function|Desc|
|-----|-----|
|`isfinite(x,/[,out,where,casting,order,...])`|逐元素是否有限|
|`isinf(x,/[,out,where,casting,order,...])`||
|`isnan(x,/[,out,where,casting,order,...])`||
|`isnat(x,/[,out,where,casting,order,...])`|逐元素是否`NaT`|
|`fabs(x,/[,out,where,casting,order,...])`|绝对值|
|`signbit(x,/[,out,where,casting,order,...])`|*signbit*是否设置，即`<0`|
|`copysign(x1,,x2,/[,out,where,casting,order,...])`|根据`x1`设置`x2`的*signbit*|
|`nextafter(x1,,x2,/[,out,where,casting,order,...])`|`x1`朝向`x2`的下个浮点数，即变动最小精度|
|`spacing(x,/[,out,where,casting,order,...])`|`x`和最近浮点数距离，即取值的最小精度|
|`modf(x[,out1,out2],/[,out,where],...)`|返回取值的整数、小数部分|
|`ldexp(x1,x2,/[,out,where,casting,...])`|计算`x1*2**x2`，即还原2为底的科学计数|
|`frexp(x[,out1,out2],/[,out,where],...)`|返回2为底的科学计数的假数、指数|
|`floor(x,/,out,*,where,...)`||
|`ceil(x,/,out,*,where,...)`||
|`trunc(x,/,out,*,where,...)`||









