---
title: Universal Functions
categories:
  - Python
  - Numpy
tags:
  - Python
  - Numpy
  - NDArray
  - Ufunc
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
	-	numpy中包含超过60种*ufunc*
		-	部分*ufunc*在相关运算标记调用时，会被自动调用

###	内部缓冲

-	Internal Buffers
	-	用于数据非对齐、数据交换、数据类型转换场合
	-	`.setbufsize(size)`：基于线程设置内部缓冲，缺省为
		`10,000`元素

###	类型转换规则

-	各*ufunc*内部维护列表，给出适用的输入类型（组合）、
	相应的输出类型
	（可通过`.types`属性查看）

-	当*ufunc*内部列表中没有给定的输入类型组合时，则需要
	进行safely类型转换
	（可通过`np.can_cast`函数判断）
	-	`"S", "U", "V"`类型不能支持*ufunc*运算
	-	标量-数组操作使用不同类型转换规则确保标量不会降低
		数组精度，除非标量和数组属于同一类型体系

###	UFunc维度说明

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

###	UFunc原型

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

##	*UFunc*相关函数

|Function|Desc|
|-----|-----|
|`apply_along_axis(func1d,axis,arr,*args,...)`|沿给定轴应用函数|
|`apply_over_axes(func,a,axes)`|依次沿给定轴应用函数`func(a,axis)`|
|`frompyfunc(func,nin,nout[,identity])`|创建ufunc，指定输入、输出数量|
|`vertorize(pyfunc[,otypes,doc,excluded,cache,signature])`|创建ufunc，较`frompyfunc`提供更多特性|
|`piecewise(x,condlist,funclist,*args,**kw)`|按照`condlist`中索引，对应应用`funclist`中函数|



