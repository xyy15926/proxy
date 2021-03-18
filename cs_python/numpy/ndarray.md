---
title: NDArray
categories:
  - Python
  - Numpy
  - NDArray
tags:
  - Python
  - Numpy
  - NDArray
  - Data Science
date: 2021-01-31 15:49:17
updated: 2021-01-31 15:49:17
toc: true
mathjax: true
description: Numpy模块中`ndarray`类
---

## NDArray

```python
class ndarray(shape[,dtype,buffer,offset])
```

-	`ndarray`：具有相同类型、大小（固定大小）项目的多维容器
	-	`ndarray`由计算中内存连续的一维段组成，并与将`N`个整数
		映射到块中项的位置的索引方案相结合
    -	可以共享相同数据段，即可以是其他数据区的视图
		- 另一个`ndarray`
		- 实现`buffer`的对象

-	属性
    - `shape`：指定尺寸、项目数量
    - `dtype`（*data-type object*）：指定项目类型
    - `strides`：存储各维度步幅，用于计算连续数据段中偏移

<https://www.numpy.org.cn/reference/arrays/ndarray.html></https://www.numpy.org.cn/reference/arrays/ndarray.html>

###	Broadcast 广播规则

Broadcasting：4条广播规则用于处理不同shape的数组

-	非维数最大者在`shape`前用`1`补足
-	输出的`shape`中各维度是各输入对应维度最大值
-	各输入的维度同输出对应维度相同、或为`1`
-	输入中维度为`1`者，对应的（首个）数据被用于沿该轴的
	所有计算
	（即对应的`stride`为`0`，*ufunc*不step along该维度）

```
shape(3, 2, 2, 1) + shape(1, 3)
	-> shape(3, 2, 2, 1) + shape(1, 1, 1, 3)
	-> shape(3, 2, 2, 3) + shape(1, 1, 2, 3)
	-> shape(3, 2, 2, 3) + shape(1, 2, 2, 3)
	-> shape(3, 2, 2, 3) + shape(3, 2, 2, 3)
```

##	数组属性

###	内存布局 

|属性|描述|
|-----|-----|
|`ndarray.flags`|有关数组内存布局的信息|
|`ndarray.shape`|数组维度（元组）|
|`ndarray.strides`|遍历数组时每个维度中的字节数量（元组）|
|`ndarray.ndim`|数组维数|
|`ndarray.data`|Python缓冲区对象指向数组的数据的开头|
|`ndarray.size`|数组中的元素数|
|`ndarray.itemsize`|数组元素的长度，以字节为单位|
|`ndarray.nbytes`|数组元素消耗的总字节数|
|`ndarray.base`|如果内存来自其他对象，则为基础对象|

###	数据类型

|属性|描述|
|-----|-----|
|[`ndarray.dtype`](https://www.numpy.org.cn/reference/arrays/dtypes.html)|元素数据类型|

###	其他属性

|属性|描述|
|-----|-----|
|`ndarray.T`|转置|
|`ndarray.real`|实数部分|
|`ndarray.imag`|虚数部分|
|`ndarray.flat`|数组的一维迭代器|

###	数组接口

|属性|描述|
|-----|-----|
|`__array_interface__`|数组接口python端|
|`__array_struct__`|数组接口C语言端|

###	`ctypes`外部函数接口

|属性|描述|
|-----|-----|
|`ndarray.ctypes`|简化数组和`ctypes`模块交互的对象|

##	`np.nditer`

-	`ndarray`对象的默认迭代器是序列类型的默认迭代器
	-	即以对象本身作为迭代器时，默认行为类似

		```python
		for i in range(X.shape[0]):
			pass
		```

|Routine|Function Version|Method Version|
|-----|-----|-----|
|`nditer(op[,flags,op_flags,...])`|高性能迭代器|无|
|`nested_iters(op,axes[,flags,op_flags,...])`|在多组轴上嵌套创建`nditer`迭代器|无|
|`ndenumerate(arr)`|`(idx,val)`迭代器|无|
|`lib.Arrayterator(var[,buf_size])`|适合大数组的缓冲迭代|
|`flat`|无|返回`np.flatiter`迭代器|
|`ndindex(*shape)`|迭代shape对应数组的索引|无|

###	`np.nditer`

```md
class np.nditer(
	op,
	flags=None,
	op_flags=None,
	op_dtypes=None,
	order='K'/'C'/'F'/'A',
	casting='safe',
	op_axes=None,
	itershape=None,
	buffersize=0
)
```

-	迭代方式
	-	通过标准python接口迭代数组中各数组标量元素
	-	显式使用迭代器本身，访问其属性、方法
		-	`np.nditer[0]`访问当前迭代的结果
		-	`np.iternext()`获取下个迭代对象

-	包含特殊属性、方法获取额外信息（可能需设置迭代标志）
	-	跟踪索引：获取索引`np.nditer.index`、
		`np.nditer.multi_index`
	-	手动迭代`np.nditer.iternext()`得到下个
		`np.nditer`对象
	-	获取操作数`np.nditer.operands`：迭代器关闭之后
		将无法访问，需要在关闭前获得引用

<https://www.numpy.org.cn/reference/arrays/nditer.html>

####	参数

-	`flags`：迭代器标志
	-	`buffered`：允许缓冲
		-	增大迭代器提供给循环内部的数据块
		-	减少开销、提升性能
	-	`c_index`：track C顺序索引
	-	`f_index`：track C顺序索引
	-	`multi_index`：track 多维索引
	-	`common_dtype`：将所有操作数转换为公共类型
		-	需设置`copying`或`buffered`
	-	`copy_if_overlap`：迭代器决定是否读操作数覆盖写
		操作数，还是使用临时副本避免覆盖
	-	`delay_bufalloc`：延迟缓冲区设置直至`reset()`函数
		调用
		-	允许`allocate`操作数在其值被复制到缓冲区前初始化
	-	`external_loop`：迭代一维数组而不是零维数组标量
		-	利于矢量化操作
		-	返回的循环块与迭代顺序相关
	-	`grow_inner`：允许迭代数组大小大于缓冲区大小
		-	`buffered`、`external_loop`均设置情况下
	-	`ranged`：
	-	`refs_ok`：允许迭代引用类型，如`object`数组
	-	`reduce_ok`：允许迭代广播后的`readwrite`操作数
		（也即`reduction`操作数）
	-	`zerosize_ok`：允许迭代大小为0

-	`op_flags`
	-	`readonly`：操作数只能被读取
	-	`readwrite`：操作数能被读写
	-	`writeonly`：操作只能被写入
	-	`no_broadcast`：禁止操作数被广播
	-	`contig`：强制操作数数据连续
	-	`aligned`：强制操作数数据对齐
	-	`nbo`：强值操作数数据按原生字节序
	-	`copy`：允许临时只读拷贝
	-	`updateifcopy`：允许临时读写拷贝
	-	`allocate`：允许数组分配若`op`中包含`None`
		-	迭代器为`None`分配空间，不会为非空操作数分配
			空间，即使是广播后赋值空间不足
		-	操作数中`op`中`None`对应`op_flags`缺省为
			`["allocate", "writeonly"]`
	-	`no_subtype`：阻止`allocate`操作数使用子类型
	-	`arraymask`：表明对应操作数为mask数组
		-	用于从设置有`writemasked`标志的操作数中选择写回
			部分
	-	`writemasked`：只有`arraymask`操作数选择的元素被写回
	-	`overlap_assume_elementwise`：标记操作数只能按照迭代
		顺序获取
		-	允许在`copy_if_overlap`设置的场合，更保守的拷贝

-	`op_dtypes`：操作数需求的数据类型
	-	在循环内对单个值进行数据类型转换效率低
	-	迭代器以缓冲、复制整体进行类型转换提高效率
	-	需要同时设置`"copy"`或`"buffered"`，否则因无法复制、
		缓冲报错（类型不同时）
		（类型转换不修改原数组值，需要额外空间存储转换后值）

-	`order`：迭代顺序
	-	`C`/`F`：C风格、Fortran风格
	-	`A`：若所有数组均为Fortran风格则为Fortran风格，否则
		为C风格
	-	`K`：尽量贴近内存布局

	> - `allocate`操作数的内存布局会兼容此参数设置

-	`casting`：指明在拷贝、缓冲时允许的数据类型转换规则
	（包括读取、写回数组时可能的类型转换）
	-	`no`：不允许任何类型转换
	-	`equiv`：仅允许字节顺序改变
	-	`safe`：仅允许可保证数据精度的类型转换
	-	`same_kind`：只能允许`safe`或同类别类型转换
	-	`unsafe`：允许所有类型转换

-	`op_axes`：设置迭代器维度到操作数维度的映射
	-	需为每个操作数设置维度映射

-	`itershape`：设置迭代器的shape

-	`buffersize`：设置缓冲区大小
	-	`buffered`设置的情况下
	-	`0`表示默认大小

####	使用说明

-	控制迭代顺序
	-	设置`order`参数
	-	缺省按照**内存布局**迭代
		-	提高效率
		-	适合不关心迭代顺序场合

	```python
	# 二者迭代顺序完全相同
	np.nditer(X, order="K")
	np.nditer(X.T)
	# 指定按C或Fortran顺序
	np.nditer(X, order="C")
	np.nditer(X, order="F")
	```

-	修改数组值
	-	设置`writeonly`、`readwrite`
		-	生成可写的缓冲区数组，并在迭代完成后复制回原始
			数组
		-	发出迭代结束信号，将缓冲区数据复制回原始数组
			-	支持`with`语句上下文管理
			-	迭代完成后手动`.close()`
	-	可设置`allocate`标志支持为空操作数分配空间
		-	对`None`参数`op`，其`op_flags`缺省设置为
			`["allocate", "readwrite"]`

		```python
		with np.nditer(X, op_flags=["readwrite"]) as it:
			for x in it:
				x[...] = 0
		```

-	迭代一维数组而不是数组标量
	-	缺省返回最低维维度长的一维数组
	-	可以通过设置`buffered`扩大返回的数组长度
		-	`buffersize`设置`buffered`大小，可用此参数决定
			返回的数组长度
		-	返回数组长度完全由`buffersize`决定，与数组shape
			无关
	```md
	a = np.arange(30).reshape(5,6)
	for x in np.nditer(a, flags=["external_loop", "buffered"], buffersize=11):
		print(x, type(x))
	```

-	跟踪、获取索引

	```python
	it = np.nditer(a, flags=["multi_index"])
	while not it.finished:
		print(it[0], it.multi_index)
		it.iternext()
	```

-	以特定数据类型迭代
	-	`op_dtypes`参数设置迭代返回的数据类型
	-	需同时设置`"copy"`或`"buffered"`字段

	```python
	for x in np.nditer(a, op_dtypes=["complex128"]):
		print(np.sqrt(x), end=" ")
	```

-	迭代器分配空间
	-	`allocate`标志表示允许为操作数分配空间，即允许空
		操作数
	-	若分配空间初值被使用，注意迭代前初始化
		（如*reduction*迭代场合）

	```md
	def square(a, ret=None):
		with np.nditer([a, ret],
			op_flags=[["readonly"], ["writeonly", "allocate"]]
		) as it:
			for x, y in it:
				y[...] = x**2
		return ret
	```

-	外积（笛卡尔积）迭代
	-	设置`op_axes`参数指定各操作数`op`各维度位置、顺序
		-	迭代器负责将迭代器维度映射回各操作数维度
		-	类似于手动自由广播

		```python
		# 指定维度位置、顺序
		it = np.nditer([a,b,None], flags=["external_loop"],
				op_axes=[[0,-1,-1], [-1,0,1],None])
		# 迭代得到外积
		with it:
			for x,y,z in it:
				z[...] = x*y
			result = it.operands[2]
		```

-	*Reduction*迭代
	-	触发条件：**可写的**操作数中元素数量**小于**迭代空间
		-	`"reduce_ok"`需被设置
		-	`"readwrite"`而不是`"writeonly"`被设置，即使循环
			内部未读
		-	暗含`"no_broadcast"`必然不被设置

	```python
	ret = np.array([0])
	with np.nditer([a,b], flags=["reduce_ok", "external_loop"],
			op_flags=[["readonly"], ["readwrite"]]) as it:
		for x,y in it:
			y[...] += x
	# 或者同样设置`allocate`标志，并且在迭代器内设置初始值
	np.nditer([a, None], flags=["reduce_ok", "external_loop"],
			op_flags=[["readonly"], ["readwrite", "allocate"]],
			op_axes=[None, [0,1,-1]])
	with it:
		# 设置初始值
		it.operands[1][...] = 0
		for x, y in it:
			y[...] += x
		result = it.operands[1]
	```

###	`nested_iters`

-	`nested_iters`：按维度嵌套`nditer`
	-	迭代参数类似`nditer`

	```python
	i, j = np.nested_iters(X, flags=["multi_index"])
	for x in i:
		print(i.multi_index)
		for y in j:
			print("", j.multi_index, y)
	```

###	`flat`迭代器

-	`X.flat`：返回C-contiguous风格迭代器`np.flatiter`
	-	支持切片、高级索引
	-	实质上是数组的一维视图

###	`np.ndenumerate`

-	`np.ndenumerate`：多维索引迭代器，返回多维索引、值元组

	```python
	for multi_idx, val in np.ndenumerate(X):
		pass
	```

###	`np.broadcast`

-	`np.broadcast`：返回（多个）数组广播结果元组的迭代器
	-	类似广播后*zip*，即先将数组广播，然后将广播后元素
		组合成元组作为迭代器中元素

	```python
	for item in np.broadcast([[1,2],[3,4]], [5,6]):
		pass
	```

