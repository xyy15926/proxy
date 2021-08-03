---
title: 数据模型--基本数据类型
categories:
  - Python
  - Py3Ref
tags:
  - Python
  - Py3Ref
  - Data Model
  - Variable
  - Object
  - Value
  - Datatype
date: 2019-06-05 11:05:43
updated: 2021-08-02 11:44:48
toc: true
mathjax: true
comments: true
description: 数据模型--基本数据类型
---

##	对象、值、类型

对象：python中对数据的抽象

-	python中所有数据都是由对象、对象间关系表示

	-	按冯诺依曼“存储程序计算机”，代码本身也是由对象表示

###	编号、类型、值

每个对象都有各自**编号**、**类型**、**值**

-	编号：可以视为对象在内存中地址，对象创建后不变

	-	`id()`函数：获取代表对象编号的整形
	-	`is`算符：比较对象编号判断是否为同一对象

-	类型：决定对象支持的操作、可能取值

	-	类型会影响对象行为几乎所有方面，甚至对象编号重要性
		也受到影响，如：对于会得到新值的运算
		-	不可变类型：可能返回同类型、同取值现有对象引用
			-	`a = b = 1`：`a`、`b`可能指向相同对象`1`
				（取决于具体实现）
		-	可变类型：不允许返回已存在对象
			-	`c=[];d=[]`：会保证`c`、`d`指向不同、单独
				空列表（`c=d=[]`将同一对象赋给`c`、`d`）
	-	对象创建后保持不变
	-	`type`：返回对象类型

	> - CPython：相同整形值都引用同一个对象

-	值：通过一些特征行为表征的抽象概念

	-	对象值在python中是抽象概念
		-	对象值没有规范的访问方法
		-	不要求具有特定的构建方式，如：值由其全部数据
			属性组成

	-	对象值可变性由其类型决定
		-	可变的：值可以改变的对象
		-	不可变的：值（直接包含对象编号）不可改变的对象

	> - 比较运算符实现了**特定对象值概念**，可以认为是
		通过实现对象比较间接定义对象值

> - CPython：`id(x)`返回存放`x`的地址

###	对象销毁

对象不会被显式销毁（`del`仅是移除名称绑定）

-	无法访问时**可能**被作为垃圾回收
	-	允许具体实现推迟垃圾回收或完全省略此机制
	-	实现垃圾回收是质量问题，只要可访问对象不会被回收
		即可
	-	不要依赖不可访问对象的立即终结机制，应当总是显式
		关闭外部资源引用

-	以下情况下，正常应该被回收的对象可能继续存活
	-	使用实现的跟踪、调试功能
	-	通过`try...except...`语句捕捉异常

> - CPython：使用带有（可选）**延迟检测循环链接垃圾**的
	引用计数方案
> > -	对象**不可访问**时立即回收其中大部分，但不保证
		回收包含**循环引用**的垃圾

###	标准类型层级结构

> - 以下是python内置类型的列表，扩展模块可以定义更多类型
> - 以下有些类型有特殊属性，这些特殊属性不应用作通常使用，
	其定义在未来可能改变

##	`None`

`NoneType`：只有一种取值，`None`是具有此值的唯一对象

-	通过内置名称`None`访问
-	多数情况表示空值，如
	-	未显式指明返回值函数返回`None`
-	逻辑值：假

##	`NotImplemented`

`NotImplementedType`：只有一种取值，`NotImplemented`是具有
此值的唯一对象

-	通过内置名称`NotImplemented`访问
-	数值、富比较方法在操作数没有该实现操作时应返回此值
	-	返回`NotImplemented`前，解释器会依据运算符尝试反射
		方法、委托回退方法
-	逻辑值：真

##	`Ellipsis`

`ellipsis`：只有一种取值，`Ellipsis`是具有此值的唯一对象

-	通过字面值`...`、内置名称`Ellipsis`访问
-	逻辑值：真

##	`numbers.Number`

`number.Number`：由数字字面值创建，被作为算法运算符、算数
内置函数返回结果

-	不可变：一旦创建其值不再改变
-	类似数学中数字，但也受限于计算机对数字的表示方法

###	`numbers.Integral`

`numbers.Integral`：表示数学中整数集合

-	`int`：整形，表示**任意大小数字，仅受限于可用内存**
	-	变换、掩码运算中以二进制表示
	-	负数以2的补码表示（类似符号位向左延伸补满空位）

-	`bool`：布尔型，表示逻辑值真、假
	-	`True`、`False`是唯二两个布尔对象
	-	整形子类型：在各类场合中行为类似整形`1`、`0`，仅在
		转换为字符串时返回`"True"`、`"False"`

####	方法、函数

-	`int.bit_length()`：不包括符号位、开头0位长
-	`int.to_bytes(length, byteorder, *, signed=False)`
-	`class int.from_bytes(bytes, byteorder, *, signed=False)`

> - 详细说明参见<https://docs.python.org/zh-cn/3/library/stdtypes.html#additional-methods-on-integer-types>

###	`numbers.Real(float)`

`float`：表示机器级**双精度浮点数**

-	接受的取值返回、溢出处理取决于底层结构、python实现
-	python不支持单精度浮点

> - 没必要因为节省处理器、内存消耗而增加语言复杂度

####	特殊取值

```python
infty = float("inf")
neg_infty = float("-inf")
	# 正/负无穷大
nan = float("nan")
	# Not a Number
```

-	特殊取值根据定义`==`、`is`肯定返回`False`
	-	`float.__eq__`内部应该有做检查，保证`==`返回`False`
	-	每次会创建“新”的`nan/infty`

	> - 连续执行`id(float("nan"))`返回值可能相等，这是因为
		每次生成的`float("nan")`对象被回收，不影响

-	`np.nan is np.nan`返回`True`，应该是`numpy`初始化的时候
	创建了一个`float("nan")`，每次都是使用同一个*nan*

####	相关操作

> - 详细参考<https://docs.python.org/zh-cn/3/library/stdtypes.html#additional-methods-on-float>
> - 更多数字运算参考`math`、`cmath`模块

-	`float.as_integer_ratio()`
-	`float.is_integer()`
-	`float.hex()`
-	`classmethod float.fromhex(s)`
-	`round(f[,n])`
-	`math.trunc(f)`
-	`math.floor(f)`
-	`math.ceil(f)`

###	`numbers.Complex(complex)`

`complex`：以一对机器级双精度浮点数表示复数值

-	实部、虚部：可通过只读属性`z.real`、`z.imag`获取

##	*Iterators*

迭代器类型

-	迭代器对象需要自身支持以下两个方法，其共同组成迭代器协议
	-	`iterator.__iter__()`
	-	`iterator.__next__()`

> - 方法详细参考*cs_python/py3ref/cls_special_method*

###	*Generator*

生成器类型：提供了实现迭代器协议的便捷形式

-	将容器对象的`__iter__()`方法实现为生成器，方便实现容器对
	迭代器支持

> - 创建、使用参见*cs_python/py3ref/dm_gfuncs*

##	序列

序列：表示以非负整数作为索引的**有限有序集**

-	不可变序列类型：对象一旦创建不能改变
	-	若包含其他可变对象引用，则可变对象“可改变”
	-	但不可变对象所**直接引用的对象集是不可变的**
	-	包括
		-	`str`
		-	`tuple`
		-	`bytes`
		-	`range`：非基本序列类型

-	可变序列：创建后仍可被改变值
	-	`list`
	-	`bytesarray`

###	通用序列操作

-	`x in s`、`x not in s`
	-	`str`、`bytes`、`bytearray`支持子序列检测

-	`s + t`：拼接
	-	拼接不可变总会生成新对象
	-	重复拼接构建序列的运行时开销将基于序列总长度乘方

-	`s * n`、`n * s`：`s`自身拼接`n`次
	-	`n<0`被当作`0`处理
	-	`s`中项不会被复制，而是被多次引用

-	`s[i]`、`s[i:j]`、`s[i:j:step]`
	-	`i<0`索引为负值：索引顺序相对于序列`s`末尾，等价于
		对序列长度取模
	-	序列切片：与序列类型相同的新序列
		-	索引从0开始
		-	左闭右开
	-	某些序列支持`a[i:j:step]`扩展切片

-	`s.index(x[, i[, j]])`
	-	仅部分序列支持
	-	类似`s[i:j].index(x)`，但返回值是相对序列开头

-	`s.count(x)`：序列中元素`x`数目

-	`len(s)`：返回序列条目数量

-	`min(s)`、`max(s)`：序列最小、最大值

> - 序列比较运算默认实现参见*cs_python/py3ref/expressions*
> - 以上运算自定义实现参见
	*cs_python/py3ref/cls_special_methods*

####	不可变序列

不可变序列普遍实现而可变序列未实现的操作

-	`hash()`内置函数

####	可变序列

-	`s[i]=x`、`s[i:j]=t`、`s[i:j:k]=t`：下标、切片被赋值
	-	`s[i:j:k]=t`中`t`长度必须和被替换切片长度相同
-	`del s[i:j]`、`del s[i:j:k]`：移除元素
	-	作为`del`语句的目标
	-	等同于`s[i:j]=[]`
-	`s.append()`：添加元素
	-	等同于`s[len(s):len(s)] = [x]`
-	`s.clear()`：移除所有项
	-	等同于`del s[:]`
-	`s.copy()`：浅拷贝
	-	等同于`s[:]`
-	`s.extend(t)`：扩展（合并）序列
	-	基本上等于`s += t`
-	`s.insert(i, x)`：向序列中插入元素
	-	等同于`s[i:i] = [x]`
-	`s.pop(i=-1)`：弹出序列中元素
-	`s.remove(x)`：删除序列中首个值为`x`的项
-	`s.reverse()`：反转序列
	-	反转大尺寸序列时，会原地修改序列
	-	为提醒用户此操作通过间接影响进行，不会返回反转后序列

> - `array`、`collections`模块提供额外可变序列类型
> - 可利用`collections.abc.MutableSequence`抽象类简化自定义
	序列操作

###	`tuple`

元组

-	元组中条目可以是任意python对象
-	元组创建
	-	一对圆括号创建空元组
	-	逗号分隔
		-	单项元组：后缀逗号`a,`、`(a,)`
		-	多项元组：`a,b,c`、`(a,b,c)`
	-	内置构建器：`tuple`、`tuple(iterable)`

###	`list`

列表

-	列表中条目可以是任意python对象
-	构建方式
	-	方括号括起、项以逗号分隔：`[]`、`[a]`、`[a,b]`
	-	列表推导式：`[x for x in iterable]`
	-	类型构造器：`list(iterable)`

####	相关操作

#####	`.sort`

```python
def list.sort(*, key=None, reverse=False):
	pass
```

-	用途：对列表**原地排序**
	-	使用`<`进行各项之间比较
	-	不屏蔽异常：若有比较操作失败，整个排序操作将失败，
		此时列表可能处于**部分被修改状态**

-	参数
	-	`key`：带参数函数，遍历处理每个元素提取比较键
		-	`None`：默认，直接使用列表项排序

-	说明
	-	`.sort`保序，有利于多重排序
	-	为提醒用户此方法原地修改序列保证空间经济性，其不返回
		排序后序列（可考虑使用`sorted`显式请求）

> - CPython：列表排序期间尝试改变、检测会造成未定义影响，
	CPython将列表排序期间显式为空，若列表排序期间被改变将
	`raise ValueError`

###	`str`

```python
class str(object="")
	# 返回`object.__str__()`、`object.__repr__()`
class str(object=b"", encoding="utf-8", errors="strict")
	# 给出`encoding`、`errors`之一，须为bytes-like对象
	# 等价于`bytes.decode(encoding, errors)`
```

字符串：由Unicode码位值组成不可变序列（应该是*UTF16-bl*编码）

-	范围在`U+0000~U+10FFFF`内所有码位值均可在字符串中使用
-	不存在单个“字符”类型
	-	字符串中单个字符为长度为1字符串
-	不存在可变字符串类型
	-	可以用`str.join()`、`io.StringIO`高效连接多个字符串
		片段
-	字符串构建
	-	字符串字面值：*cs_python/py3ref/lexical_analysis*
	-	内置构造器`str()`

####	相关操作

> - 参见<https://docs.python.org/zh-cn/3/library/stdtypes.html#string-methods>

-	`ord()`：转换单个字符字符串为（整形）码位
-	`chr()`：转换（整形）码位为单个字符字符串

#####	判断

-	`str.isalnum()`
-	`str.isalpha()`
-	`str.isascii()`
-	`str.isdecimal()`
-	`str.isdigit()`
-	`str.isidentifier()`
-	`str.islower()`
-	`str.isnumeric()`
-	`str.isprintable()`
-	`str.isspace()`
-	`str.istitle()`
-	`str.isupper()`

#####	查找

-	`str.rfind(sub[, start[, end]])`
-	`str.rindex(sub[, start[, end]])`
-	`str.startswith(prefix[, start[, end]])`
-	`str.endwith(suffix[, start[, end]])`
-	`str.count(sub[, start[, end]])`：子串出现次数
-	`str.find(sub[, start[, end]])`
	-	仅检查`sub`是否为子串，应使用`in`
	-	找不到子串时返回`-1`
-	`str.index(sub[, start[, end]])`
	-	类似`str.find`，但找不到子串时`raise ValueError`

#####	分隔

-	`str.partition(sep)`
-	`str.rpartition(sep)`
-	`str.rsplit(sep=None, maxsplit=-11)`
-	`str.split(sep=None, maxsplit=-1)`
-	`str.splitline([keepends])`

#####	拼接

-	`str.join(iterable)`
-	`str.strip([chars])`
-	`str.lstrip([chars])`
-	`str.rstrip([chars])`
-	`str.rstrip([chars])`

#####	转换

-	`str.lower()`
-	`str.upper()`
-	`str.swapcase()`
-	`str.translate(table)`
-	`str.replace(old, new[, count])`
-	`static str.maketrans(x[, y[, z]])`
-	`str.encode(encoding="utf-8", errors="strict")`：使用
	指定编码方案编码为`bytes`
-	`str.expandtabs(tabsize=8)`
-	`str.capitalize()`：首字符大写副本
-	`str.casefold()`：消除大小写副本
-	`str.center(width[, fillchar])`：字符串位于中间的字符串
-	`str.title()`

#####	格式化

-	`str.ljust(width[, fillchar])`
-	`str.rjust(width[, fillchar])`
-	`str.zfill(width)`
-	`str.format(*args, **kwargs)`
-	`str.format_map(mapping)`
	-	类似`str.format(**mapping)`，但`mapping`不会被复制
		到`dict`中
		```python
		class Default(dict):
			def __missing__(self, key):
				return key
		"{name} was born in {country}".format_map(Default(name="Guido"))
		```

#####	*printf*风格字符串格式化

-	`format % values`中：`format`中`%`转换标记符将被转换
	为`values`中条目

	-	效果类似于`sprintf`
	-	`values`为与`format`中指定转换符数量等长元组、或映射
		对象，除非`format`要求单个参数

-	转换标记符按以下顺序构成
	-	`%`字符：标记转换符起始
	-	映射键：可选，圆括号`()`括起字符序列
		-	`values`为映射时，映射键必须
	-	转换旗标：可选，影响某些类型转换效果
		-	`#`：值转换使用“替代形式”
		-	`0`：为数字值填充`0`字符
		-	`-`：转换值左对齐（覆盖`0`）
		-	` `：符号位转换产生整数（空字符串）将留出空格
		-	`+`：符号字符显示在开头（覆盖` `）
	-	最小字段宽度：可选
		-	`*`：从`values`读取下个元素
	-	精度：可选，`.`之后加精度值
		-	`*`：从`values`读取下个元素
	-	长度修饰符：可选
	-	转换类型
		-	`d`/`u`/`i`：十进制整形
		-	`o`：8进制整形
			-	`#`替代形式，前端添加`0o`
		-	`x`/`X`：小/大写16进制整形
			-	`#`替代形式，前端添加`0x/0X`
		-	`e`/`E`：小/大写浮点指数
			-	`#`替代形式，总是包含小数点
		-	`f`/`F：浮点10进制
			-	`#`替代形式，总是包含小数点
		-	`g`/`G`：指数小于-4、不小于精度使用指数格式
			-	`#`替代形式，总是包含小数点，末尾`0`不移除
		-	`c`：单个字符（接收整数、单个字符字符串）
		-	`r`/`s`/`a`：字符串（`repr`/`str`/`ascii`转换）
			-	按输出精度截断
		-	`%`：输出`%`字符

####	技巧

-	快速字符串拼接
	-	构建包含字符串的列表，利用`str.join()`方法
	-	写入`io.StringIO`实例，结束时获取值

###	`bytes`/`bytearray`

```python
class bytes([source[, encoding[, errors]]])
```

> - 字节串：单个字节构成的不可变序列
> - 字节数组：字节串可变对应版本，其他同不可变`bytes`

-	字节串构建
	-	字节串字面值：*cs_python/py3ref/lexical_analysis*
	-	内置构造器`bytes()`
		-	指定长度零值填充：`bytes(10)`
		-	整数组成可迭代对象：`bytes(range(20))`
		-	通过缓冲区协议复制现有二进制数据：`bytes(obj)`

-	字节数组构建
	-	字节数组没有字面值语法，只能通过构造器构造
	-	可变，构建空字节数组有意义

-	类似整数构成序列
	-	每个条目都是8位字节
	-	取值范围`0~255`，但只允许ASCII字符*0~127*
	-	`b[0]`产生整数，切片返回`bytes`对象
	-	可通过`list(bytes)`将`bytes`对象转换为整数构成列表

> - 由`memeoryview`提供支持

####	相关函数、方法

-	`bytes.decode`：解码为相关字符串
-	`classmethod bytes.fromhex(string)`
-	`bytes.hex()`

> - 其他类似字符串，包括*printf*风格格式化
> - 参见<https://docs.python.org/zh-cn/3/library/stdtypes.html#bytes-and-bytearray-operations>

####	技巧

-	快速字节串拼接
	-	构建包含字节串的列表，利用`bytes.join()`方法
	-	写入`io.BytesIO`实例，结束时获取值
	-	使用`betaarray`对象进行原地拼接

###	`memoryview`

```python
class memoryview(obj)
```

内存视图：允许python代码访问对象内部数据

-	若对象支持缓冲区协议，则无需拷贝
	-	支持缓冲区协议的内置对象包括`bytes`、`bytesarray`

-	内存视图元素：原始对象`obj`处理的基本内存单元
	-	对简单`bytes`、`bytesarray`对象，一个元素就是一字节
	-	`array.array`等类型可能有更大元素

-	内存视图支持索引抽取、切片
	-	若下层对象可选，则支持赋值，但切片赋值不允许改变大小

####	相关操作

> - 参见<https://docs.python.org/zh-cn/3/library/stdtypes.html#memory-views>

-	`mv.__eq__(exporter)`
-	`mv.__len__()`
-	`mv.tobyte()`
-	`mv.hex()`
-	`mv.tolist()`
-	`mv.release()`
-	`mv.cast(format[, shape])`：将内存视图转换新格式形状

####	可用属性

以下属性均只读

-	`mv.obj`：内存视图的下层对象
-	`mv.nbytes`
	-	`== product(shape) * itemsize = len(mv.tobytes())`
-	`mv.readonly`
-	`mv.format`：内存视图中元素格式
	-	表示为`struct`模块格式
-	`mv.itemsize`
-	`mv.ndim`
-	`mv.shape`
-	`mv.strides`
-	`mv.suboffsets`
-	`mv.c_contiguous`
-	`mv.f_contiguous`
-	`mv.contiguous`

###	*Slices Object*

切片对象：表示`__getitem__()`方法得到的切片

-	可以使用内置的`slice()`函数创建
-	`a[start: stop]`形式的调用被转换为
	`a[slice(start, stop, None)]`

> - 切片对象是内部类型，参见*cs_python/py3ref/dm_exec*，也
	不是序列类型

####	特殊只读属性

-	`start`：下界
-	`stop`：上界
-	`step`：步长值

> - 属性可以具有任意类型

####	方法

-	`.indices(self, length)`：计算切片对象被应用到`length`
	长度序列时切片相关信息
	-	返回值：`(start, stop, step)`三元组
	-	索引号缺失、越界按照正规连续切片方式处理

###	`range`

`range`：不可变数字序列类型（非不是基本序列类型）

```python
class range(stop)
class range(start=0, stop[, step=1])
```

-	参数：必须均为整数（`int`或实现`__index__`方法）
	-	`step > 0`：对range对象`r[i]=start + step * i`，其中
		`i >= 0, r[i] < stop`
	-	`step < 0`：对range对象`r[i]=start + step * i`，其中
		`i >= 0, r[i] > stop`
	-	`step = 0`：`raise ValueError`

-	说明
	-	允许元素绝对值大于`sys.maxsize`，但是某些特性如：
		`len()`可能`raise OverflowError`
	-	`range`类型根据需要计算单项、切片值
		-	相较于常规`list`、`tuple`占用内存较小，且和表示
			范围大小无关
		-	只能表示符合严格模式的序列
	-	`range`类型实现了`collections.abc.Sequence`抽象类
		-	基本实现序列所有操作：检测、索引查找、切片等
		-	除拼接、重复：拼接、重复通常会违反严格模式
	-	`!=`、`==`将`range`对象视为序列比较，即提供相同值即
		认为相等

##	集合类型

-	表示**不重复**、**不可变**对象组成的无序、有限集合
	-	不能通过下标索引
	-	可以迭代
	-	可以通过内置函数`len`返回集合中条目数量

-	常用于
	-	快速成员检测、去除序列中重复项
	-	进行交、并、差、对称差等数学运算

###	公用操作

> - 参见<https://docs.python.org/zh-cn/3/library/stdtypes.html#set-types-set-frozenset>

-	`len(s)`
-	`x [not ]in s`
-	`s.isdisjoint(other)`
-	`s.issubset(other)`/`s <= other`
-	`s < other`
-	`s.issuperset(other)`/`s >= other`
-	`s > other`
-	`s.union(*others)`/`s | other |...`
-	`s.intersection(*others)`/`s & other &...`
-	`s.difference(*other)`/`s - other - other`
-	`s.symmetric_difference(other)`/`s ^ other`
-	`s.copy()`

> - 集合比较仅定义偏序，集合列表排序无意义

####	可变集合独有

-	`s.update(*others)`/`s |= other |...`
-	`s.intersection_update(*others)`/`s &= other &...`
-	`s.difference_udpate(*others)`/`s -= other |...`
-	`s.symmetric_difference_update(other)`/`set ^= other`
-	`s.add(elem)`
-	`s.remove(elem)`
-	`s.discard(elem)`
-	`s.pop()`
-	`s.clear()`

###	`set`/`frozenset`

```python
class set([iterable])
class frozenset([iterable])
```

> - 集合：由具有唯一性的*hashable*对象组成的多项无序集
> - 冻结集合：不可变集合，可哈希，可以用作集合元素、字典键

-	创建集合
	-	`set()`内置构造器
	-	花括号包括、逗号分隔元组列表：`{a, b}`

-	创建冻结集合
	-	`frozenset()`内置构造器

-	python中集合类似`dict`通过hash实现
	-	集合元素须遵循同字典键的不可变规则
	-	数字：相等的数字`1==1.0`，同一集合中只能包含一个

####	操作说明

-	`.remove`、`.__contains__`、`discard`等可以接收`set`类型
	参数，其将被转换为临时`frozenset`对象

-	非运算符版本操作可以接受任意可迭代对象作为参数，运算符
	版本只能接受集合类型作为参数

##	映射

映射：表示任何索引集合所索引的对象的集合

-	通过下标`a[k]`可在映射`a`中选择索引为`k`的条目
	-	可在表达式中使用
	-	可以作为赋值语句、`del`语句的目标

> - `dbm.ndbm`、`dbm.gnu`、`collections`模块提供额外映射类型

###	通用操作

> - 参见<https://docs.python.org/zh-cn/3/library/stdtypes.html#mapping-types-dict>

-	`len(d)`
-	`d[key]`
-	`key [not ]in d`
-	`iter(d)`
-	`d.keys()`：返回字典视图对象
-	`d.values()`：返回字典视图对象
-	`d.items()`：返回字典视图对象
-	`d.get(key[, default])`
-	`d.copy()`
-	`classmethod fromkey(iterable[, value])`

####	可变映射独有

-	`d[key]=value`
-	`del d[key]`
-	`d.clear()`
-	`d.setdefault(key[, default])`
-	`d.pop()`
-	`d.popitem()`
-	`d.copy()`
-	`d.update()`

###	`dict`

```python
class dict(**kwargs)
class dict(mapping, **kwargs)
class dict(iterable, **kwargs)
```

字典：可由**几乎任意值作为索引**的有限个对象可变集合

-	字典的高效实现要求使用键hash值以保持一致性
	-	不可作为键的值类型
		-	包含列表、字典的值
		-	其他通过对象编号而不是值比较的可变对象
	-	数字：相等的数字`1==1.0`索引相同字典条目

-	创建字典
	-	花括号括起、逗号分隔键值对：`{key:value,}`
	-	内置字典构造器：`dict()`

####	字典视图对象

字典视图对象：提供字典条目的**动态视图**，随字典改变而改变

-	`len(dictview)`
-	`iter(dictview)`
-	`x in dictview`

> - 参见<https://docs.python.org/zh-cn/3/library/stdtypes.html#dictionary-view-objects>

