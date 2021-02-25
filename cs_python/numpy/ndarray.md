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

##	数组方法

> - 类标准数学运行方法`ndarray`均有实现，参加类方法

###	数组转换

|方法|描述|
|-----|-----|
|`.item(*args)`|复制元素至标准python标量|
|`.tolist()`|转换为`.ndim`层嵌套python标量列表|
|`.itemset(*args)`|插入元素（尝试转换类型）|
|`.tobytes([order])`/`.tostring`|转换为字节串|
|`.tofile(fid[,sep,format])`|作为文本或二进制写入文件|
|`.dump(file)`|以pickle格式存储至文件|
|`.dumps()`|返回pickle字节串|
|`.astype(dtype[,order,casting,...])`|转换为指定类型|
|`.byteswap([inplace])`|交换字节|
|`.copy([order])`|复制|
|`.view([dtype,type])`|创建新视图|
|`.getfield(dtype[,offset])`|设置数据类型为指定类型|
|`.setflags([write,align,uic])`|设置标志|
|`.fill(value)`|使用标量填充|

###	形状操作

|方法|描述|
|-----|-----|
|`.reshape(shape[,order])`|创建新形状数组|
|`.resize(new_shape[,refcheck])`|直接改变数组形状|
|`.transpose(*axes)`|返回轴转置视图|
|`.swapaxes(axis1,axis2)`|返回轴交换视图|
|`.flatten([order])`|创建一维副本|
|`.ravel([order])`|返回一维视图|
|`.squeeze([axis])`|删除长度为1的维度|

###	项目选择、操作

|方法|描述|
|-----|-----|
|`.take(indices[,axis,out,mode])`|获取指定位置元素|
|`.put(indices,values[,mode])`|设置元素值|
|`.repeat(repeats[,axis])`|重复数组元素|
|`.choose(choices[,out,mode])`|以自身为索引从`choices`中选择元素|
|`.sort([axis,kind,order])`||
|`.argsort([axis,kind,order])`|返回对数组排序的索引|
|`.partition(kth[,axis,kind,order])`|用`kth`元素作为基准快排划分|
|`.argpartition(kth[,axis,kind,order])`|返回快排划分的索引|
|`.searchsorted(v[,side,sorter])`|元素`v`应插入的位置|
|`.compress(condition[,axis,out])`|返回选定切片|
|`.diagonal([offset,axis1,axis2])`|返回对角线|

###	计算

-	`axis=None`：默认值`None`，表示在整个数组上执行操作

|方法|描述|
|-----|-----|
|`.max([axis,out,keepdims,initial,...])`||
|`.argmax([axis,out])`||
|`.min([axis,out,keepdims,initial,...])`||
|`.argmin([axis,out])`||
|`.ptp([axis,out,keepdims])`|极差|
|`.clip([min,max,out])`|以`min-max`过滤|
|`.conj()`|共轭|
|`.round([decimals,out])`|四舍五入|
|`.trace([offset,axis1,axis2,dtype,out])`|迹|
|`.sum([axis,dtype,out,keepdims,...])`||
|`.cumsum([axis,dtype,out])`|累计和|
|`.mean([axis,dtype,out,ddof,keepdims])`||
|`.var([axis,dtype,out,ddof,keepdims])`||
|`.std([axis,dtype,out,ddof,keepdims])`||
|`.prod([axis,dtype,out,keepdims])`||
|`.cumprod([axis,dtype,out])`||
|`.all([axis,out,keepdims])`|与|
|`.any([axis,out,keepdims])`|或|

##	索引、切片

###	基本切片、索引

-	基本切片`[Slice]start:stop:step`（基本同原生类型切片）
	-	`start`、`stop`负值时，按维度长取正模
	-	`step>0`时，`start`缺省为`0`、`stop`缺省为维度长`N`
	-	`step<0`时，`start`缺省为`N-1`、`stop`缺省为`-N-1`
	-	`stop`、`start`可以超过维度长`N`

-	`Ellipsis`/`...`：放在切片中表示选择所有
	-	`...`存在的场合，结果总是数组而不是数组标量，即使其
		没有大小

-	`np.newaxis`/`None`：为切片生成数组在所在位置添加长度为
	`1`的维度

-	切片可以用于设置数组中的值

> - 基本切片可认为是依次对各维度切片，若靠前维度为索引，则
	可以把靠前维度独立出来
> - 基本切片生成的所有数组始终是原始数组的视图，也因此存在
	切片引用的数组内存不会被释放
> - 注意：基本索引可用于改变数组的值，但是返回值不是对数组
	中对应值的引用

###	高级索引

-	选择对象为以下类型时会触发高级索引
	-	非元组序列
	-	`ndarray`（整形或boolean类型）
	-	包含至少一个序列、`ndarray`（整型或boolean类型）的
		元组

-	高级索引总是返回数据的**副本**

####	整数索引

-	整数索引`X[obj]`允许根据其各维度索引选择数组`X`任意元素
	-	各整数索引（数组）表示对应维度的索引
	-	各维度索引迭代、连接得到各元素位置：`zip(obj*)`
	-	索引维数小于数组维数时，以子数组作为元素
		（可以理解为索引和数组高维对齐后广播）

-	整数索引结果shape由`obj`中各维度索引shape决定
	-	整数索引`obj`中各维度索引数组会被广播
		-	各维度索引shape可能不同
		-	为保证各维度索引能正常迭代选取元素，各维度索引
			shape需要能被广播、符合广播要求
	-	则高级索引出现场合
		-	“普通索引（标量值）”不存在，必然被广播
		-	切片能够共存

-	切片（包括`np.newaxis`）和高级索引共存时
	-	高级索引特点导致其结果维度不可割
		-	“标量索引”本应削减该维度
		-	而高级索引整体（广播后）决定唯一shape
	-	高级索引结果维度应整体参与结果构建
		-	高级索引被切片分割：高级索引结果维度整体提前
		-	高级索引相邻：高级索引结果维度填充至该处

> - 高级索引操作结果中无元素，但单个维度索引越界的错误未定义
> - 高级索引结果内存布局对每个索引操作有优化，不能假设特定
	内存顺序

```python
X = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
rows = [0, 3]
cols = [0, 2]
 # 整数索引
X[np.ix_(rows, cols)]
 # 整数索引数组
X[[[1,2],[2,1]],:]
X.take([[1,2],[2,1]], axis=0)
```

####	Boolean索引

-	Boolean索引`obj`选择其中`True`处位置对应元素
	-	索引`obj`维数较数组`X`小，直接抽取子数组作为元素
		（可以理解为索引和数组高维对齐后广播）
	-	索引`obj`在超出数组`X.shape`范围处有`True`值，会引发
		索引错误
	-	索引`obj`在`X.shape`内未填充处等同于填充`False`

-	Boolean索引通过`.nonezero`方法转换为高级整数索引实现
	-	Boolean索引等价于`True`数量长的1维整数索引
		-	`X[..,bool_obj,..]`等价于
			`X[..,bool_obj.nonzero(),..]`
		-	Boolean索引总是削减对应索引，展开为1维
	-	Boolean索引、高级整数索引共同存在场合行为诡异
		-	Boolean索引转换为等价的整数索引
		-	整数索引需要广播兼容转换后整数索引
		-	整数索引、转换后整数索引整体得到结果

> - 索引`obj`和数组`X`形状相同计算速度更快

###	字段名称形式访问

-	`ndarray`中元素为结构化数据类型时，可以使用字符串索引
	访问
	-	字段元素非子数组时
		-	其shape同原数组
		-	仅包含该字段数据
		-	数据类型为该字段数据类型
	-	字段元素为子数组时
		-	子数组shape会同原数组shape合并
	-	支持字符串列表形式访问
		-	返回数组视图而不是副本（Numpy1.6后）

##	NDArray标量

-	numpy中定义了24种新python类型（NDArray标量类型）
	-	类型描述符主要基于CPython中C语言可用的类型

-	标量具有和`ndarray`相同的属性和方法
	-	数组标量不可变，故属性不可设置

![numpy_dtype_hierarchy](imgs/numpy_dtype_hierarchy.png)

###	内置标量类型

-	*Booleans*

	|类型|说明|字符代码|
	|-----|-----|-----|
	|`bool_`|兼容python `bool`|`'?'`|
	|`bool8`||

	> - `bool_`不是`int_`子类，而python3中`bool`继承自`int`

-	*Integers*

	|类型|说明|字符代码|
	|-----|-----|-----|
	|`byte`|兼容C `char`|`'b'`|
	|`short`|兼容C `short`|`'h'`|
	|`intc`|兼容C `int`|`'i'`|
	|`int_`|兼容python `int`|`'l'`|
	|`longlong`|兼容C `long`|`'q'`|
	|`intp`|大到足以适合指针|`'p'`|
	|`int8`|8位||
	|`int16`|16位||
	|`int32`|32位||
	|`int64`|64位||

	> - `int_`不是继承自python3内置`int`

-	*Unsigned Integers*

	|类型|说明|字符代码|
	|-----|-----|-----|
	|`ubyte`|兼容C `unsigned char`|`'B'`|
	|`ushort`|兼容C `unsigned short`|`'H'`|
	|`uintc`|兼容C `unsigned int`|`'I'`|
	|`uint_`|兼容python `int`|`'L'`|
	|`ulonglong`|兼容C `long`|`'Q'`|
	|`uintp`|大到足以适合指针|`'P'`|
	|`uint8`|8位|`'I1'`|
	|`uint16`|16位|`'I2'`|
	|`uint32`|32位|`'I4'`|
	|`uint64`|64位|`'I8'`|

-	*Floating-Point Numbers*

	|类型|说明|字符代码|
	|-----|-----|-----|
	|`half`||`'e'`|
	|`single`|兼容C `float`|`'f'`|
	|`double`|兼容C `double`||
	|`float_`|兼容（继承自）python `float`|`'d'`|
	|`longfloat`|兼容C `long float`|`'g'`|
	|`float16`|16位|`'f2'`|
	|`float32`|32位|`'f4'`|
	|`float64`|64位|`'f8'`|
	|`float96`|96位（需平台支持）||
	|`float128`|128位（需平台支持）||

-	*Complex Floating-Point Numbers*

	|类型|说明|字符代码|
	|-----|-----|-----|
	|`csingle`|单精度复数|`'F'`|
	|`complex_`|兼容（继承自）python `complex`|`'D'`|
	|`clongfloat`||`'G'`|
	|`complex64`|双32位复数||
	|`complex128`|双64位复数||
	|`complex192`|双96位复数||
	|`complex256`|双128位复数||

-	*Any Python Object*

	|类型|说明|字符代码|
	|-----|-----|-----|
	|`object_`|任何python对象|`'O'`|

	> - 实际存储是python对象的引用，其值不必是相同的python
		类型

-	*Flexible*

	|类型|说明|字符代码|
	|-----|-----|-----|
	|`bytes_`|兼容（继承自）python `bytes`|`'S#'`/`'a#'`|
	|`unicode_`|兼容（继承自）python `unicode/str`|`'U#'`|
	|`void`||`'V#'`|

> - 数组标量类型代码字符和数据类型中类型字符不完全一致，而是
	兼容`struct`模块等

###	属性

-	数组标量属性基本同`ndarray`

###	索引

-	数组标量类似0维数组一样支持索引

###	方法

-	数组标量与`ndarray`有完全相同的方法
	-	默认行为是在内部将标量转换维等效0维数组，并调用相应
		数组方法

###	定义数组标量类型

-	从内置类型组合结构化类型
-	子类化`ndarray`
	-	部分内部行为会由数组类型替代
-	完全自定义数据类型，在numpy中注册
	-	只能使用numpy C-API在C中定义

##	数据类型`np.dtype`

```python
class dtype(obj[,align,copy])
```

-	`numpy.dtype`类描述如何解释数组项对应内存块中字节
	-	数据大小
	-	数据内存顺序：*little-endian*、*big-endian*
	-	数据类型
		-	结构化数据
			-	各字段名称
			-	各字段数据类型
			-	字段占用的内存数据块
		-	子数组
			-	形状
			-	数据类型

> - 数组标量类型不是`numpy.dtype`实例，即使有些场合可用其
	替代数据类型对象（标量类型内蕴结构）
> - numpy中函数、方法需要数据类型参数时，可提供`dtype`实例，
	或可转换为`dtype`的对象（即可用于创建`dtype`的参数）

###	数据类型元素

####	类型类

-	NumPy内置类型
	-	24中内置数组标量类型
	-	泛型类型

		|Generic类型|转换后类型|
		|-----|-----|
		|`number`,`inexact`,`floating`|`float`|
		|`complexfloating`|`cfloat`|
		|`integer`,`signedinteger`|`int_`|
		|`unsignedinteger`|`uint`|
		|`character`|`string`|
		|`generic`,`flexible`|`void`|

-	python内置类型，等效于相应数组标量
	-	`None`：缺省值，转换为`float_`
	-	其他内置类型

		|Python内置类型|转换后类型|
		|-----|-----|
		|`int`|`int_`|
		|`bool`|`bool_`|
		|`float`|`float_`|
		|`complex`|`cfloat`|
		|`bytes`|`bytes`|
		|`str`|`unicode_`|
		|`unicode`|`unicode_`|
		|`buffer`|`void`|
		|Others|`object_`|

-	带有`.dtype`属性的类型：直接访问、使用该属性
	-	该属性需返回可转换为`dtype`对象的内容

####	可转换类型的字符串

-	`numpy.sctypeDict.keys()`中字符串

-	*Array-protocal*类型字符串
	-	首个字符指定数据类型
	-	（可选）其余字符指定项目字节数
		（只能指定满足平台要求的字节数）
		（*Unicode*指定字符数）

	|代码|类型|
	|-----|-----|
	|`'?'`|boolean|
	|`'b'`|(signed) byte，等价于`'i1'`|
	|`'B'`|unsigned byte，等价于`'u1'`|
	|`'i'`|(signed) integer|
	|`'u'`|unsigned integer|
	|`'f'`|floating-point|
	|`'c'`|complex-floating point|
	|`'m'`|timedelta|
	|`'M'`|datetime|
	|`'O'`|(Python) objects|
	|`'S'`/`'a'`|zero-terminated bytes (not recommended)|
	|`'U'`|Unicode string|
	|`'V'`|raw data (void)|

> - 与数组标量描述字符不完全一致

###	结构化数据类型

-	结构化数据类型
	-	包含一个或多个数据类型字段，每个字段有可用于访问的
		名称
	-	父数据类型应有足够大小包含所有字段
	-	父数据类型几乎总是基于`void`类型

-	仅包含不具名、单个基本类型时，数组结构会穿透
	-	字段不会被隐式分配名称
	-	子数组shape会被添加至数组shape

####	参数格式

-	可转换数据类型的字符串指定类型、shape
	-	依次包含四个部分
		-	字段shape
		-	字节序描述符：`<`、`>`、`|`
		-	基本类型描述符
		-	数据类型占用字节数
			-	对非变长数据类型，需按特定类型设置
			-	对变长数据类型，指字段包含的数量
	-	逗号作为分隔符，分隔多个字段
	-	各字段名称只能为默认字段名称

	> - 对变长类型，仅设置shape时，会将其视为bytes长度

	```python
	dt = np.dtype("i4, (2,3)f8, f4")
	```

-	元组指定字段类型、shape
	-	元组中各元素指定各字段名、数据类型、shape：
		`(<field_name>, <dtype>, <shape>)`
		-	若名称为`''`空字符串，则分配标准字段名称
	-	可在列表中多个元组指定多个字段
		`[(<field_name>, <dtype>, <shape>),...]`
	-	数据类型`dtype`可以**嵌套其他数据类型**
		-	可转换类型字符串
		-	元组/列表

	```python
	dt = np.dtype(("U10", (2,2)))
	dt = np.dtype(("i4, (2,3)f8, f4", (2,3))
	dt = np.dtype([("big", ">i4"), ("little", "<i4")])
	```

-	字典元素为名称、类型、shape列表
	-	分别指定名称列表、类型列表等：
		`{"names":...,"formats":...,"offsets":...,"titles":...,"itemsize":...}`
		-	`"name"`、`"formats"`为必须
		-	`"itemsize"`指定总大小，必须足够大
	-	分别指定各字段：`"field_1":..., "field_2":...`
		-	不鼓励，容易与上一种方法冲突

	```python
	dt = np.dtype({
		"names": ['r', 'g', 'b', 'a'],
		"formats": ["u1", "u1", "u1", "u1"]
	})
	```

-	解释基数据类型为结构化数据类型：
	`(<base_dtype>, <new_dtype>)`
	-	此方式使得`union`成为可能

	```python
	dt = np.dtype(("i4", [("r", "I1"), ("g", "I1"), ("b", "I1"), ("a", "I1")]))
	```

###	属性

-	描述数据类型

	|属性|描述|
	|-----|-----|
	|`.type`|用于实例化此数据类型的数组标量类型|
	|`.kind`|内置类型字符码|
	|`.char`|内置类型字符码|
	|`.num`|内置类型唯一编号|
	|`.str`|类型标识字符串|

-	数据大小

	|属性|描述|
	|-----|-----|
	|`.name`|数据类型位宽名称|
	|`.itemsize`|元素大小|

-	字节顺序

	|属性|描述|
	|-----|-----|
	|`.byteorder`|指示字节顺序|

-	字段描述

	|属性|描述|
	|-----|-----|
	|`.fields`|命名字段字典|
	|`.names`|字典名称列表|

-	数组类型（非结构化）描述

	|属性|描述|
	|-----|-----|
	|`.subtype`|`(item_dtype,shape)`|
	|`.shape`||

-	附加信息

	|属性|描述|
	|-----|-----|
	|`.hasobject`|是否包含任何引用计数对象|
	|`.flags`|数据类型解释标志|
	|`.isbuiltin`|与内置数据类型相关|
	|`.isnative`|字节顺序是否为平台原生|
	|`.descr`|`__array_interface__`数据类型说明|
	|`.alignment`|数据类型需要对齐的字节（编译器决定）|
	|`.base`|基本元素的`dtype`|

###	方法

-	更改字节顺序

	|方法|描述|
	|-----|-----|
	|`.newbyteorder([new_order])`|创建不同字节顺序数据类型|

-	Pickle协议实现

	|方法|描述|
	|-----|-----|
	|`.reduce()`|pickle化|
	|`.setstate()`||

##	迭代数组

-	`ndarray`对象的默认迭代器是序列类型的默认迭代器
	-	即以对象本身作为迭代器时，默认行为类似

		```python
		for i in range(X.shape[0]):
			pass
		```

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

###	`flat`迭代器

-	`X.flat`：返回C-contiguous风格迭代器
	-	支持切片、高级索引

> - `X.flat`实际上是数组的一维视图

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

