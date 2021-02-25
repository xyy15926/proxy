---
title: NDArray子类
categories:
  - Python
  - Numpy
  - NDArray
tags:
  - Python
  - Numpy
  - NDArray
  - Data Science
date: 2021-02-18 17:10:18
updated: 2021-02-18 17:10:18
toc: true
mathjax: true
description: NumPy模块中`ndarray`子类介绍，基本包括
  掩码类、记录类等
---

##	子类相关钩子属性、方法

###	`__array__`方法

-	`class.__array_ufunc__(ufunc, method, *inputs, **kwargs)`
	-	功能：供自定义以覆盖numpy中ufunc行为
		-	返回操作结果，或`NotImplemented`
			（将此方法置`None`）
	-	参数
		-	`ufunc`：被调用的ufunc对象
		-	`method`：字符串，指示调用的`ufunc`对象的方法
		-	`inputs`：`ufunc`顺序参数元组
		-	`kwargs`：`ufunc`关键字参数字典

	> - *Ufunc*、与`__array_ufunc__`关系参见ufunc部分

-	`class.__array_function__(func,types,args,kwargs)`
	-	参数
		-	`func`：任意callable，以`func(*args, **kwargs)`
			形式调用
		-	`types`：来自实现`
		-	`args`、`kwargs`：原始调用的参数

-	`class.__array__finalize(obj)`
	-	功能：构造之后更改`self`的属性
		-	在为`obj`类型数组分配空间时调用
	-	参数
		-	`obj`：`ndarray`子类

-	`class.__array_prepare__(array,context=None)`
	-	功能：在ufunc计算前，将ouput数组转换为子类实例、
		更新元数据
		-	调用任何ufunc前，在最高优先级的input数组，或指定
			的output数组上调用，返回结果传递给ufunc
		-	默认实现：保持原样

-	`class.__array_wrap__(array,context=None)`
	-	功能：在将结果返回给用户前，将output数组转换为子类
		实例、更新元信息
		-	ufunc计算结果返回给用户前，在最高优先级的output
			数组、或指定output对象上调用
		-	默认实现：将数组转换为新

-	`class.__array__([dtype])`
	-	功能：若output对象有该方法，ufunc结果将被写入其
		返回值中

> - 若ufunc中所有`__array_ufunc__`返回`NotImplemented`，那么
	`raise TypeError`

###	`__array__`属性

-	`class.__array_priority__`
	-	功能：决定返回对象的数据类型（有多种可能性时）
		-	默认值：`0.0`


##	`np.matrix`

-	Matrix对象继承自`ndarray`，具有相同的属性、方法，除
	-	可使用Matlab样式的字符串表示法创建Matrix对象
		-	空格分割列
		-	`;`分割行
	-	始终二维
		-	`.ravel()`仍然二维
		-	*item selection*返回二维对象
	-	覆盖乘法为矩阵乘法
	-	覆盖幂次为矩阵幂次
	-	默认`__array_priority__`为`10.0`
	-	特殊方法

		|Method|Desc|
		|-----|-----|
		|`matrix.T`|转置|
		|`matrix.H`|复数共轭|
		|`matrix.I`|逆矩阵|
		|`matrix.A`|返回`ndarray`|

-	Matrix创建方法

	|Method|Desc|
	|-----|-----|
	|`np.matrix(data[,dtype,copy])`|不建议使用|
	|`np.asmatrix(data[,dtype])`|将数据转换为矩阵|
	|`np.bmat(obj[,ldict,gdict])`|从字符串、嵌套序列、数组中构建|


> - 主要用于与`scipy.sparse`交互，建议不使用
> - `np.mat`是`np.matrix`别名

##	`np.memmap`

-	`np.memmap`：内存映射文件数组，使用内存映射文件作为数组
	数据缓冲区
	-	对大文件，使用内存映射可以节省大量资源

-	方法

	|Method|Desc|
	|-----|-----|
	|`np.memmap(filename[,dtype,mode,shape])`|创建存储在磁盘文件的内存映射数组|
	|`np.flush()`|flush内存数据至磁盘|

##	`np.char`

-	`np.char`：`string_`、`unicode_`数据类型的增强型数组
	-	继承自`ndarray`
	-	继承由Numarray引入的特性：项检索和比较操作中，数组
		元素末尾空格被忽略
	-	定义有基于元素的`+`、`*`、`%`的操作
	-	有所有标准`string`、`unicode`方法

-	类、方法、函数

	|Method|Desc|
	|-----|-----|
	|`np.chararray(shapep[,itemsize,unicode,...])`|提供string或unicode值的方便视图|
	|`np.core.defchararray.array(obj[,itemsize,...])`|创建chararray|

> - `chararray`类是为了后向兼容Numarray，建议使用`object_`、
	`string_`、`unicode_`类型、`numpy.char`中的自由函数用于
	字符串快速向量化操作

##	`np.rec`

-	`np.rec`：允许将结构化数组的字段作为属性访问

-	类、方法、函数

	|Method|Desc|
	|-----|-----|
	|`np.recarray`|创建允许属性访问字段的`ndarray`|
	|`np.record`|允许使用属性查找字段的数据类型标量|

##	`np.ma`

-	掩码数组：由标准`np.ndarray`和掩码组成
	-	掩码可以是*nomask*或指示元素是否有效的*boolean*数组

-	`np.ma`模块核心是`MasekdArray`类（`ndarray`子类）

> - <https://www.numpy.org.cn/reference/arrays/maskedarray.html>

###	`np.ma.MaskedArray`

####	创建掩码数组

-	直接调用类`MaskedArray`（或其别名`masked_array`）
-	调用构造方法`array(data[,dtype,copy,order,mask...]`
-	在已有数组基础上创建视图
-	以下一些方法

	|Method|Desc|
	|-----|-----|
	|`asrray(a[,dtype,order])`|将输入`a`转换为给定类型的掩码数组|
	|`asanyarray(a[,dtype])`|转换为掩码数组，保留子类类型|
	|`fix_invalid(a[,mask,copy,fill_value])`|转换为掩码数组，并替换无效值|
	|`masked_equal(x,value[,copy])`|mask值与`value`相同部分|
	|`masked_greater(x,value[,copy])`||
	|`masked_greater_equal(x,value[,copy])`||
	|`masked_not_equal(x,value[,copy])`||
	|`masked_less(x,value[,copy])`||
	|`masked_less_equal(x,value[,copy])`||
	|`masked_inside(x,v1,v2[,copy])`|mask值在`[v1,v2]`区间内部分|
	|`masked_outside(x,v1,v2[,copy])`|mask值在`[v1,v2]`区间外部分|
	|`masked_invalid(a[,copy])`|mask无效值（`NaN`、`inf`）|
	|`masked_values(x,value[,rtol,atol,copy])`|使用浮点数规则判断等于|
	|`masked_where(condition,a[,copy])`|mask满足`condition`部分|

####	访问掩码数组

-	访问数据
	-	通过`.data`属性：可能是`ndarray`或其子类的视图
		-	等同于直接在掩码数组上创建`ndarray`或其子类视图
	-	`__array__`方法：`ndarray`
	-	使用`ma.getdata`函数

-	访问掩码mask
	-	通过`.mask`属性
	-	通过`ma.getmask`、`ma.getmaskarray`函数

-	仅访问有效数据
	-	对掩码mask取反作为索引`~X.mask`
	-	使用`.compressed`方法得到一维`ndarray`

	```python
	print(X[~X.mask])
	print(X.compressed())
	```

####	修改掩码数组

-	mask元素
	-	给元素赋特殊值`ma.masked`
	-	直接修改`.mask`属性（不推荐）

-	unmask元素
	-	给元素赋有效值
		-	被设置`hardmask`时会失败

-	设置mask是否为hard
	-	`ma.array`中`hard_mask`参数指定
	-	`.harden_mask()`、`.soften_mask()`方法设置

> - 可设置`.mask=True/False`等设置掩码数组整体mask
> - `ma.nomask`是`np.bool_`类型的`False`，`ma.masked`是特殊
	常数

####	索引、切片

-	索引非结构化掩码数组
	-	mask为`False`：返回数组标量
	-	mask为`True`：返回`ma.masked`

-	索引结构化掩码数组
	-	所有字段mask均为`False`：返回`np.void`对象
	-	存在字段mask为`True`：返回零维掩码数组

-	切片
	-	`.data`属性：原始数据视图
	-	`.mask`属性：`ma.nomask`或者原始mask视图

###	`np.ma`运算

-	掩码数组支持代数、比较运算
	-	无效元素不参与运算，元素在运算前后保持不变

-	`np.ma`模块中对大部分ufunc有特别实现
	-	对于定义域有限制的一元、二元运算，无定义的结果会
		自动mask

	```python
	ma.log([-1, 0, 1, 2])
	```

-	掩码数组也支持标准的ufunc，返回掩码数组
	-	运算中任意元素被mask，则结果中相应元素被mask
	-	若ufunc返回可选的上下文输出，则上下文会被处理，且
		无定义结果被mask

##	标准容器类

-	`np.lib.user_array.container`
	-	为向后兼容、作为标准容器类而引入
	-	其中`self.array`属性是`ndarray`
	-	比`ndarray`本身更方便多继承

-	类、方法、函数

	|Method|Desc|
	|-----|-----|
	|`np.lib.user_array.container(data[,...])`|简化多继承的标准容器类|




