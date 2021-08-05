---
title: NDArray开发
categories:
  - Python
  - Numpy
tags:
  - Python
  - Numpy
  - NDArray
  - Data Science
date: 2021-02-19 09:02:01
updated: 2021-02-19 09:02:01
toc: true
mathjax: true
description: 
---

##	NDArray Interface/Protocol

-	数组接口（规范）：为重用数据缓冲区设计的规范
	-	接口描述内容
		-	获取`ndarray`内容的方式
		-	数组需为同质数组，即其中各元素数据类型相同
	-	接口包含C和Python两个部分
		-	Python-API：对象应包含属性`__array_interface__`字典
		-	C-API：结构体`__array_struct__`

<https://www.numpy.org.cn/en/reference/arrays/interface.html#python-side>

###	Python API

> - `__array_interface__`：由3个必须字段和5个可选字段构成

-	`shape`：各维度长度（使用时注意取值范围）

-	`typestr`：指明同质数组数据类型的字符串
	-	格式、含义基本同*Array-Protocol*，但有部分字符
		含义不同
	-	但不同于自定义数据类型字符串，不指定结构化数据、
		shape，非基本类型就是`void`，具体含义由`descr`
		给出

	|代码|类型|
	|-----|-----|
	|`'t'`|bit|
	|`'b'`|boolean|
	|`'B'`|unsigned byte|
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

-	`descr`：给出同质数组中各元素中内存布局的详细描述的
	列表
	-	各元素为包含2、3个元素的元组
		-	名称：字符串、或`(<fullname>,<basicname>)`
			形式的元组
		-	类型：描述基础类型字符串、或嵌套列表
		-	shape：该结构的重复次数，若没有给出则表示无
			重复
	-	一般此属性在`typestr`为取值为`V[0-9]+`时使用，
		要求表示的内存字节数相同
	-	缺省为`[(''), typestr]`

-	`data`：给出数据位置的2元素元组或暴露有缓冲接口
	的对象
	-	元组首个元素：表示存储数组内容的数据区域，指向
		数据中首个元素（即`offset`被忽略）
	-	元素第二个元素：只读标记
	-	缺省为`None`，表示内存共享通过缓冲接口自身实现，
		此时`offset`用于指示缓冲的开始

-	`strides`：存储各维度跃迁的strides的元组
	-	元组各元素为各维度跃迁字节数整形值，注意取值范围
	-	缺省为`None`，C-contiguous风格

-	`mask`：指示数据是否有效的暴露有缓冲接口的对象
	-	其shape需要同原始数据shape广播兼容
	-	缺省为`None`，表示所有数据均有效

-	`offset`：指示数组数据区域offset的整形值
	-	仅在数据为`None`或为`buffer`对象时使用
	-	缺省为`0`

-	`version`：指示接口版本

###	C API

-	`__array_struct__`：ctype的`PyCObject`，其中`voidptr`
	指向`PyArrayInterface`
	-	`PyCObject`内存空间动态分配
	-	`PyArrayInterface`有相应的析构，访问其之后需要在其上
		调用`Py_DECREF`

	```c
	typedef struct{
		int two;				// 值为2，sanity check
		int nd;					// 维数
		char typekind;			// 数组中数据类型
		int itemsize;			// 数据类型size
		int flags;				// 指示如何解释数据的标志
								// 5bits指示数据解释的5个标志位
									// `CONTIGUOUS`	0x01
									// `FROTRAN`	0x02
									// `ALIGNED`	0x100
									// `NOTSWAPPED` 0x200
									// `WRITABLE`	0X400
								// 1bit指示接口解释（是否包含有效`descr`字段）
									// `ARR_HAS_DESCR` 0x800
		Py_intptr_t *shape;		// shape
		Py_intptr_t *strides;	// strides
		void *data;				// 指向数组中首个元素
		PyObject *descr;		// NULL或数据描述（需设置`flags`中的`ARR_HAS_DESCR`，否则被忽略）
	} PyArrayInterface;
	```





