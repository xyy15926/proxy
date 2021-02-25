---
title: NDArray标量
categories:
  - Python
  - Numpy
  - NDArray
tags:
  - Python
  - Numpy
  - NDArray
  - Data Science
date: 2021-02-01 08:32:44
updated: 2021-02-01 08:32:44
toc: true
mathjax: true
description: 
---

##	标量

-	numpy中定义了24种新python类型变量
	-	标量具有和`ndarray`相同的属性和方法
		（数组标量不可变，属性不可设置）

![numpy_dtype_hierarchy](numpy_dtype_hierarchy.png)

###	内置标量类型

-	*Booleans*

	|类型|说明|
	|-----|-----|
	|`bool_`|兼容python `bool`|
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
	|`intp|大到足以适合指针|`'p'`|
	|`int8|8位||
	|`int16|16位||
	|`int32|32位||
	|`int64|64位||

	> - `int_`不是继承自python3内置`int`

-	*Unsigned Integers*

	|类型|说明|字符代码|
	|-----|-----|-----|
	|`ubyte`|兼容C `unsigned char`|`'B'`|
	|`ushort`|兼容C `unsigned short`|`'H'`|
	|`uintc`|兼容C `unsigned int`|`'I'`|
	|`uint_`|兼容python `int`|`'L'`|
	|`ulonglong`|兼容C `long`|`'Q'`|
	|`uintp|大到足以适合指针|`'P'`|
	|`uint8|8位||
	|`uint16|16位||
	|`uint32|32位||
	|`uint64|64位||

-	*Floating-Point Numbers*

	|类型|说明|字符代码|
	|-----|-----|-----|
	|`half`||`'e'`|
	|`single`|兼容C `float`|`'f'`|
	|`double`|兼容C `double`||
	|`float_`|兼容（继承自）python `float`|`'d'`|
	|`longfloat`|兼容C `long float`|`'g'`|
	|`float16`|16位||
	|`float32`|32位||
	|`float64`|64位||
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

-	*Any Python Object*及

	|类型|说明|字符代码|
	|-----|-----|-----|
	|`object_`|任何python对象|`'O'`|
	|`bytes_`|兼容（继承自）python `bytes`|`'S#'`|
	|`unicode_`|兼容（继承自）python `unicode/str`|`'U#'`|
	|`void`||`'V#'`|

	> - 实际存储是python对象的引用，其值不必是相同的python
		类型

###	属性

-	数组标量属性基本同`ndarray`

###	索引

-	数组标量类似0维数组一样索引

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

##	Datetime

-	Numpy种时间相关数据类型
	-	支持大量时间单位
	-	基于POSIX时间存储日期时间
	-	使用64位整形存储值，也由此决定了时间跨度

> - <https://www.numpy.org.cn/reference/arrays/datetime.html>

###	`np.datetime64`

-	`np.datetime64`表示单个时刻
	-	若两个日期时间具有不同单位，可能仍然代表相同时刻
	-	从较大单位转换为较小单位是安全的投射

####	创建

-	创建规则
	-	内部存储单元自动从字符串形式中选择单位
	-	接受`"NAT"`字符串，表示“非时间”值
	-	可以强制使用特定单位

-	基本方法：ISO 8601格式的字符串

	```python
	np.datetime64("2020-05-23T14:23")
	np.datetime64("2020-05-23T14:23", "D")
	```

-	从字符串创建日期时间数组

	```python
	np.array(["2020-01-23", "2020-04-23"], dtype="datetime64")
	np.array(["2020-01-23", "2020-04-23"], dtype="datetime64[D]")
	np.arange("2020-01-01", "2020-05-03", dtype="datetime64[D]")
	```

> - `np.datetime64`为向后兼容，仍然支持解析时区

###	`np.timedelta64`

-	`np.timedelta64`：时间增量

> - `np.timedelta64`是对`np.datetime64`的补充，弥补Numpy对
	物理量的支持

####	创建

-	创建规则
	-	接受`"NAT"`字符串，表示“非时间”值数字
	-	可以强制使用特定单位

-	直接从数字创建

	```python
	np.timedelta64(100, "D")
	```

-	从已有`np.timedelta64`创建，指定单位
	-	注意，不能将月份及以上转换为日，因为不同时点进制不同

	```python
	np.timedelta(a, "M")
	```

####	运算

-	`np.datetime64`可以和`np.timedelta64`联合使用

	```python
	np.datetime64("2020-05-14") - np.datetime64("2020-01-12")
	np.datetime64("2020-05-14") + np.timedelta64(2, "D")
	```

###	相关方法

|Function|Desc|
|-----|-----|
|`np.busday_offset(date,offset[,roll,weekmask,holidays,busdaycal,out])`|工作日offset|
|`np.is_busday(date[,weekmask,holidays,busdaycal,out])`|判断是否是工作日|
|`np.busday_count(begindates,enddates[,weekmask,holidays,busdaycal,out])`|指定天数|

-	`np.busday_offset`中
	-	`roll`缺省为`"raise"`，要求`date`本身为工作日









