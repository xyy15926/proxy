---
title: 二进制数据服务
categories:
  - Python
  - Py3std
tags:
  - Python
  - Py3std
  - Binary
date: 2019-06-10 01:10:04
updated: 2019-06-10 01:10:04
toc: true
mathjax: true
comments: true
description: 二进制数据服务
---

##	`struct`

`struct`模块：用于打包、拆包C`struct`格式二进制数据

-	使用python字节串存储C`struct`数据
-	需要给出格式声明，指定二进制中存储格式

###	格式说明

####	字节序、对齐

|格式符|字节序（大、小端）|类型大小|字段对齐方式|
|-----|-----|-----|-----|
|`@`（缺省值）|原生字节序|原生大小|原生对齐|
|`=`|原生字节序|标准大小|标准对齐|
|`<`|lb-endian|标准大小|标准对齐|
|`>`|bl-endian|标准大小|标准对齐|
|`!`|同`>`|

> - 原生对齐：C对齐方式，字段起始位置须为其长度整数倍
> - 标准对齐：按字节对齐，无需使用`0`补齐

####	类型

|Format|C Type|Python|Bytes|
|------|------|------|-----|
|`x`|pad byte|no value|1|
|`?`|`_Bool`|`bool`|1|
|`h`|`short`|`integer`|2|
|`H`|`unsigned short`|`integer`|2|
|`i`|`int`|`integer`|4|
|`I`|`unsigned int`|`integer`|4|
|`l`|`long`|`integer`|4|
|`L`|`unsigned long`|`long`|4|
|`q`|`long long`|`long`|8|
|`Q`|`unsigned long long`|`long`|8|
|`f`|`float`|`float`|4|
|`d`|`double`|`float`|4|
|`c`|`char`|`str` of length1|1|
|`b`|`signed char`|`bytes` of length1|1|
|`B`|`unsigned char`|`bytes` of length1|1|
|`s`|`char[]`|`str`|1|
|`p`|pascal string，带长度|`str`|NA|
|`n`|`ssize_t`|`integer`|NA|
|`N`|`size_t`|`integer`|NA|
|`P`|`void *`|足够容纳指针的整形|NA|

-	在类型符前添加数字可以指定类型重复次数

-	字符、字符串类型实参须以字节串形式给出
	-	字符：必须以长度为1字节串形式给出，重复须对应多参数
	-	字符串：可以是任意长度字节串，重复对应单个字符串
		-	长于格式指定长度被截断
		-	短于格式指定长度用`\0x00`补齐

-	python按以上长度封装各类型，但C各类型长度取决于平台，
	以上近视64位机器最可能对应C类型
	-	非64位机器`long long`类型大部分为4bytes

-	用途
	-	封装、解压数据
	-	`reinterpret_cast`类型转换

###	`Struct`

```python
class Struct:
	def __init__(self, fmt):
		pass
```

-	用途：根据指定格式`fmt`编译`Sturct`对象
	-	`Sturct`对象可以调用以下方法，无需再次指定`fmt`

###	*Pack*

```python
def struct.pack(fmt, v1, v2,...) -> bytes:
	pass
def struct.pack_into(fmt, buffer, offset , v1, v2, ...):
	pass
```

-	`pack`：按照指定格式`fmt`封装数据为字节串
-	`pack_into`：将字节串写入`buffer`指定偏移`offset`处

###	*Unpack*

```python
def struct.unpack(fmt, buffer) -> (v1, v2,...):
	pass
def struct.unpack(fmt, buffer, offset=0) -> (v1, v2, ...):
	pass
def struct.iter_unpack(fmt, buffer) -> iterator(v1, v2,...):
	pass
```

-	`unpack`：按照指定格式`fmt`拆包字节串`buffer`
-	`unpack_from`：从偏移`offset`处开始拆包
-	`iter_unpack`：迭代解压包含多个`fmt`的`buffer`

###	`calsize`

```python
def struct.calsize(fmt) -> integer:
	pass
```

-	用途：计算封装指定格式`fmt`所需字节数

##	`codecs`


