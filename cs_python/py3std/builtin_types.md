---
title: Python3 内置类型
categories:
  - Python
  - Py3std
tags:
  - Python
  - Py3std
  - References
date: 2022-04-27 10:31:14
updated: 2022-07-06 11:55:10
toc: true
mathjax: true
description: 
---

##	数字类型

-	存在 3 种不同数字类型：整数、浮点、复数
	-	内置类型
		-	整数：无限精度
			-	布尔值时整数的子类型
		-	浮点：通常使用 C 中 `double` 实现
		-	复数：包括浮点表示实部、虚部
	-	创建方式
		-	数字字面值
		-	内置函数、运算符结果
	-	支持混合运算：较窄精度操作数将拓宽至另一操作数

-	标准库包含附加的数字类型，如
	-	`fractions.Fraction`：有理数
	-	`decimal.Decimal`：可定制精度的浮点数

> - `sys.float_info` 可查看浮点具体的精度、内部表示可通过

###	整数类型附加方法

|方法|含义|说明|
|----|----|----|
|`int.bit_length()`|二进制位长|不包括符号位、开头 0 位长|
|`int.bit_count()`|绝对值二进制中 1 数量||
|`int.to_bytes(length, byteorder, *, signed=False)`|整数字节表示||
|`class int.from_bytes(bytes, byteorder, *, signed=False)`|由字节创建整数||
|`int.as_integer_ratio()`||整数比率总以整数本身作为分子|

###	浮点附加方法

|方法|含义|说明|
|----|----|----|
|`float.as_integer_ratio()`|分数表示||
|`float.is_integer()`|是否可用有限位整数表示||
|`float.hex()`|16 进制字符串表示|包含前导 `0x1.`、尾随 `p+<N>`|
|`classmethod float.fromhex(s)`|从 16 进制字符串创建浮点||
|`round(f[,n])`|||
|`math.trunc(f)`|||
|`math.floor(f)`|||
|`math.ceil(f)`|||

##	序列类型

-	序列
	-	内置 3 种基本序列类型
		-	`list`
		-	`tuple`
		-	`range`
	-	特别定制的专门化序列类型
		-	`str`
		-	`bytes`、`bytearray`

###	通用序列操作

|语法|含义|说明|
|-----|-----|-----|
|`s1 == s2`|元素、顺序相同||
|`x [not ]in s`|包含|`==` 比较；`str`、`bytes`、`bytearray` 等专门化序列支持子序列检测|
|`s + t`|拼接|不可变序列拼接生成新对象|
|`s * n`、`n * s`| `s` 自拼接 `n` 次|`n<0` 被当作`0` 处理；`s`中项不会被复制，而是被多次引用|
|`s[i]`|下标选择|`i < 0` 时表示相对于序列 `s` 末尾，即对序列长度取模|
|`s[i:j]`|切片标注|左闭右开|
|`s[i:j:step]`|扩展切片标注|仅部分序列支持|
|`s.index(x[, i[, j]])`|相对于 `j` 的索引|仅部分序列支持|
|`s.count(x)`|计数||
|`len(s)`|长度||
|`min(s)`|序列最小值||
|`max(s)`|最大值||
|`hash()`|哈希|仅部分不可变序列实现|

-	说明
	-	重复拼接构建序列的运行时开销将基于序列总长度乘方
	-	切片标注
		-	与原序列类型相同
		-	可省略左、右端点，表示 `0`、剩余全部
	-	以上运算均为内置序列的默认实现，可通过自定义序列特殊方法改变逻辑
		-	`array`、`collections` 模块提供额外可变序列类型
		-	可利用 `collections.abc.MutableSequence` 抽象类简化自定义序列操作

###	不可变序列操作

|语法|含义|说明|
|-----|-----|-----|
|`hash()`|哈希|仅部分不可变序列实现|

###	可变序列操作

|语法|含义|说明|
|-----|-----|-----|
|`s[i]=x`|下标赋值||
|`s[i:j]=t`|切片赋值||
|`s[i:j:k]=t`|高级切片赋值|`t` 长度必须和被替换切片长度相同|
|`del s[i:j]`、`del s[i:j:k]`|移除元素|等同于 `s[i:j]=[]`|
|`s.append(x)`|添加元素|等同于 `s[len(s):len(s)] = [x]`|
|`s.clear()`|移除所有项|等同于 `del s[:]`|
|`s.copy()`|浅拷贝|等同于`s[:]`|
|`s.extend(t)`|扩展序列||
|`s.insert(i, x)`|插入元素|等同于 `s[i:i] = [x]`|
|`s.pop(i=-1)`|弹出元素||
|`s.remove(x)`|删除首个 `x` 值||
|`s.reverse()`|反转序列|直接修改序列，不返回|

##	通用序列

###	`list`

-	`list`：可变序列，通常用于存储同类项目的集合
	-	构建方式
		-	`[...,...]`：方括号中用 `,` 分隔项
		-	`[x for x in iterable]`：列表推导式
		-	`list()`：类型构造器
		-	其他一些操作，如：`sorted` 函数

-	`sort(self, *, key=None, reverse=False)`：原地排序
	-	一般方法
		-	`key`：带参数函数，遍历处理每个元素提取比较键，缺省使用列表项
		-	返回值：`None`（以提醒原地排序的空间经济性）
	-	说明
		-	使用 `<` 进行各项之间比较
		-	不屏蔽异常：若有比较操作失败，整个排序操作将失败，此时列表可能处于部分被修改状态
		-	稳定，即不改变比较结果相等的元素的相对顺序，有利于多重排序

> - CPython：列表排序期间尝试改变、检测会造成未定义影响，列表排序期间显示为空，若列表排序期间被改变将 `raise ValueError`

###	`tuple`

-	`tuple`：不可变序列，通常用于存储异构数据多项集
	-	构建方式
		-	`...,...`：`,` 分隔的多项，圆括号可选
			-	单项末尾须带 `,`
			-	空元组为 `()`
		-	`tuple()`：类型构造器

> - `collections.namedtuple` 命名元组可用于建立通过名称访问异构多项集

###	`range`

-	`range([start=0,]stop[,step=1]`：不可变数字序列类型（非不是基本序列类型）
	-	其参数必须均为整数（`int` 或实现 `__index__` 方法）
		-	`step > 0`：对range对象`r[i]=start + step * i`，其中
			`i >= 0, r[i] < stop`
		-	`step < 0`：对range对象`r[i]=start + step * i`，其中
			`i >= 0, r[i] > stop`
		-	`step = 0`：`raise ValueError`
	-	说明
		-	允许元素绝对值大于 `sys.maxsize`，但是某些特性如：`len` 可能 `raise OverflowError`
		-	`range` 类型根据需要计算单项、切片值
			-	相较于常规 `list`、`tuple` 占用内存较小，且和表示范围大小无关
			-	只能表示符合严格模式的序列
		-	`range` 类型实现了 `collections.abc.Sequence` 抽象类
			-	基本实现序列所有操作：检测、索引查找、切片等
			-	除拼接、重复：拼接、重复通常会违反严格模式
		-	`!=`、`==` 将 `range` 对象视为序列比较，即提供相同值即认为相等

##	文本序列

```python
class str(object="")
	# 返回`object.__str__()`、`object.__repr__()`
class str(object=b"", encoding="utf-8", errors="strict")
	# 给出`encoding`、`errors`之一，须为bytes-like对象
	# 等价于`bytes.decode(encoding, errors)`
```

###	字符类型

|方法|含义|说明|
|----|----|----|
|`str.isalnum()`||空字符串返回 `False`，下同|
|`str.isalpha()`|||
|`str.isascii()`|||
|`str.isdecimal()`|||
|`str.isdigit()`|||
|`str.isidentifier()`|是否为有效标识符||
|`str.islower()`|||
|`str.isnumeric()`|||
|`str.isprintable()`|||
|`str.isspace()`|||
|`str.istitle()`|是否各单词首字符大写||
|`str.isupper()`|||

###	查找子串

|方法|含义|说明|
|----|----|----|
|`str.rfind(sub[, start[, end])`|||
|`str.rindex(sub[, start[, end])`|||
|`str.startswith(prefix[, start[, end])`|||
|`str.endswith(suffix[, start[, end])`|||
|`str.count(sub[, start[, end]])`|子串计数||
|`str.find(sub[, start[, end])`||未找到返回 `-1`|
|`str.index(sub[, start[, end])`||类似 `find`，但未找到时 `raise ValueError`|

###	分割、拼接

|方法|含义|说明|
|----|----|----|
|`str.partition(sep)`|在 `sep` 首次位置拆分字符串得到三元组，未找到则后两项为空字符串||
|`str.rpartition(sep)`|||
|`str.rsplit(sep=None, maxsplit=-1)`|||
|`str.split(sep=None, maxsplit=-)`|||
|`str.splitline(keepends=False)`|按行边界拆分||
|`str.join(iterable)`|用 `str` 连接||
|`str.strip([chars])`|||
|`str.lstrip([chars])`|||
|`str.rstrip([chars])`|||
|`str.center(width[, fillchar])`|位于中间的子串||

###	转换

|方法|含义|说明|
|----|----|----|
|`str.lower()`|||
|`str.upper()`|||
|`str.swapcase()`|||
|`str.replace(old, new[, count])`|||
|`static str.maketrans(x[, y[, z]])`|创建转换对照表 `{<ORD>: <TARGET>}`||
|`str.translate(table)`|根据转换对照表转换字符串||
|`str.encode(encoding="utf-8", errors="strict")`|使用指定编码方案编码为 `bytes`||
|`str.expandtabs(tabsize=8)`|||
|`str.capitalize()`|创建首字符大写副本||
|`str.casefold()`|创建消除大小写副本|类似转为小写，但更彻底|
|`str.title()`|创建单词首字符大写副本||
|`ord()`|转换单个字符字符串为（整形）码位||
|`chr()`|转换（整形）码位为单个字符字符串||


###	格式化

|方法|含义|说明|
|----|----|----|
|`str.ljust(width[, fillchar])`|||
|`str.rjust(width[, fillchar])`|||
|`str.zfill(width)`|||
|`str.format(*args, **kwargs)`|||
|`str.format_map(mapping)`||类似 `str.format(**mapping)`，但 `mapping` 不会被复制到 `dict` 中|

-	说明
	-	`format_map` 适合参数为 `dict` 子类实例的情况，避免自定义功能丢失

```python
class Default(dict):
	def __missing__(self, key):					# 自定义默认行为，避免丢失
		return key
"{name} was born in {country}".format_map(Default(name="Guido"))
```

####	*printf*风格字符串格式化

-	`<FORMAT> % <VALUES>` 中：`FORMAT` 中 `%` 转换标记符将被转换为 `VALUES` 中条目
	-	效果类似于 `sprintf`
	-	`VALUES` 为与 `FORMAT` 中指定转换符数量等长元组、映射对象（除非单个参数）

-	转换标记符按以下顺序构成
	-	`%`字符：标记转换符起始
	-	映射键：可选，圆括号 `()` 括起字符序列
		-	`VALUES` 为映射时，映射键必须
	-	转换旗标：可选，影响某些类型转换效果
	-	最小字段宽度：可选
		-	`*` 表从 `VALUES` 读取下个元素
	-	精度：可选，`.` 之后加精度值
		-	`*` 表从 `VALUES` 读取下个元素
	-	长度修饰符：可选
	-	转换类型

|旗标|含义|说明|
|----|----|----|
|`#`|值转换使用“替代形式”||
|`0`|为数字值填充 `0` 字符||
|`-`|转换值左对齐|覆盖 `0`|
|` `|符号位转换产生整数（空字符串）将留出空格||
|`+`|符号字符显示在开头|覆盖 ` `|

|转换类型|含义|说明|
|----|----|----|
|`d`、`u`、`i`|十进制整形||
|`o`|8进制整形|`#` 替代形式，前端添加 `0o`|
|`x`、`X`|小、大写 16 进制整形|`#` 替代形式，前端添加 `0x`、`0X`|
|`e`、`E`|小、大写浮点指数|`#` 替代形式，总包含小数点|
|`f`、`F`|10 进制浮点|`#` 替代形式，总是包含小数点|
|`g`、`G`|指数小于 -4、或不小于精度才使用指数格式|`#` 替代形式，总包含小数点，末尾 `0` 不移除|
|`c`|单个字符|接收整数、单个字符字符串|
|`r`、`s`、`a`|字符串 `repr`、`str`、`ascii`转换|按输出精度截断|
|`%`|输出 `%` 字符||

###	技巧

-	快速字符串拼接
	-	构建包含字符串的列表，利用 `str.join()` 方法
	-	写入`io.StringIO`实例，结束时获取值

##	二进制序列

-	`bytes`、`bytearray` 是操作二进制数据的核心内置类型
	-	支持序列的通用操作
	-	支持大部分 `str` 方法，包括 *printf* 风格格式化
	-	能与任何 *bytes-like* 对象互操作

###	`bytes`

-	`bytes([source[, encoding[, errors]]])`：单个字节构成的不可变序列
	-	创建方式
		-	字面值：字面值仅允许 *ASCII* 字符
			-	码值超过 127 的字符需使用转义序列形式
		-	内置构造器
			-	`bytes(<LEN>)`：指定长度零值填充
			-	`bytes(<ITERABLE>)`：从整数迭代对象构建
			-	`bytes(<OBJ>)`：通过缓冲区协议复制现有二进制数据
	-	`bytes` 行为类似不可变的整数序列
		-	`list` 可将 `bytes` 转换为整数列表

|方法|含义|说明|
|----|----|----|
|`class bytes.fromhex(string)`|解码字符串|空白符被忽略|
|`bytes.hex(sep[,bytes_per_sep])`|16 进制格式||

###	`bytearray`

-	`bytearray([source[, encoding[, errors]]])`：单个字节构成的可变序列
	-	`bytes` 的可变对应物，基本同 `bytes`
	-	创建方式：内置构造器
		-	`bytearray(<LEN>)`：指定长度零值填充，缺省 0 长度
		-	`bytearray(<ITERABLE>)`：从整数迭代对象构建
		-	`bytearray(<OBJ>)`：通过缓冲区协议复制现有二进制数据

###	`memoryview`

-	`memoryview(obj)`：内存视图
	-	用于访问对象（需支持缓冲区协议）内部数据，且无需拷贝
	-	内存视图元素：原始对象处理的原子内存单元
		-	简单类型，如 `bytes`、`bytearray`，元素是字节
		-	其他类型，如 `array.array` 可能是更大元素


> - 缓冲区协议 <https://docs.python.org/zh-cn/3/c-api/buffer.html#bufferobjects>

####	相关操作

|方法|含义|说明|
|----|----|----|
|`memoryview.__eq__(exporter)`|||
|`memoryview.__len__()`、`len(memoryview)`|元素数量||
|`memoryview.tobyte()`|将缓冲区数据作为字节串返回||
|`memoryview.hex()`|以 16 进制格式返回缓冲区字节串对象||
|`memoryview.tolist()`|||
|`memoryview.toreadonly()`|返回只读版本||
|`memoryview.release()`|释放视图所公开的低层缓冲区|许多对象在被视图获取时操作受限|
|`memoryview.cast(format[, shape])`|将内存视图转换新格式形状|目标格式仅限 `struct` 语法中单一元素原生格式|

####	可用属性

|只读属性|含义|说明|
|----|----|----|
|`memoryview.obj`|内存视图的下层对象||
|`memoryview.nbytes`||等于 `product(shape) * itemsize = len(mv.tobytes())`|
|`memoryview.readonly`|||
|`memoryview.format`|内存视图中元素格式|表示为`struct`模块格式|
|`memoryview.itemsize`|||
|`memoryview.ndim`|多维数组维度||
|`memoryview.shape`|||
|`memoryview.strides`|字节长度表示 `shape`||
|`memoryview.suboffsets`|||
|`memoryview.c_contiguous`|||
|`memoryview.f_contiguous`|||
|`memoryview.contiguous`|||

##	集合

-	`set([iterable])`：不重复的 *hashable* 对象组成的可变无序多项集
	-	用途
		-	成员检测
		-	重复项剔除
		-	数学集合计算
	-	特点
		-	无序，不记录元素位置、插入顺序
		-	不支持索引、切片或其他序列操作
	-	构造方式
		-	`{,}`：花括号中逗号分隔
		-	`{x for x in ...}`：集合推导式
		-	`set()`：类型构造器

-	`frozenset([iterable])`：`set` 对应的可变、*hashable* 版
	-	构造方式：`frozenset()` 类型构造器

###	公用操作

|方法|钩子操作|含义|说明|
|----|----|----|----|
|`s.__len__()`|`len(s)`|||
|`s.__contains__(x)`|`[not ]in`|||
|`s.__eq__(other)`|`==`、`!=`|元素相同|||
|`s.isdisjoint(other)`||不相交||
|`s.issubset(other)`|`<=`|子集||
||`<`|真子集||
|`s.issuperset(other)`|`>=`|父集||
||`>`|真父集||
|`s.union(*others)`|`|`|并集||
|`s.intersection(*others)`|`&`|交集||
|`s.difference(*other)`|`-`|差集||
|`s.symmetric_difference(other)`|`^`|对称差集||
|`s.copy()`||||

-	说明
	-	`__contains__` 可以接收 `set` 类型参数，其将被转换为临时 `frozenset`对象
	-	集合比较仅定义偏序，集合列表排序无意义
	-	非运算符版本可接受任意可迭代对象作为参数，而运算符钩子只能接受集合对象

####	可变集合独有

|方法|钩子操作|含义|说明|
|----|----|----|----|
|`s.update(*others)`|`|=`|更新|在位并|
|`s.intersection_update(*others)`|`&=`|交集|在位交|
|`s.difference_udpate(*others)`|`-=`|差集|在位差|
|`s.symmetric_difference_update(other)`|`^=`|对称差集|在位对称差|
|`s.add(elem)`||添加元素||
|`s.remove(elem)`||移出元素||
|`s.discard(elem)`||移出元素|类似 `remove`，但不报错|
|`s.pop()`||移出任意元素||
|`s.clear()`||清空集合||

-	操作说明
	-	`remove`、`discard` 等可以接收 `set` 类型参数，其将被转换为临时 `frozenset`对象
	-	非运算符版本操作可以接受任意可迭代对象作为参数，运算符版本只接受集合

##	映射

-	映射：将 *hashable* 值映射到任意对象的可变对象
	-	目前仅有一种标准映射类型 `dict`
	-	字典键为虚伪 *hashable* 值
		-	数值类型键遵循数字比较规则，同值数字索引同意条目（即使类型不同）

-	`dict(mapping, **kwargs)`：字典
	-	创建方式
		-	`{k:v, }`：花括号中 `,` 分隔的 `:` 键值对
		-	`{k:v for k,v in ...}`：字典推导式
		-	`dict()`：类型构造器

###	`dict` 操作

|方法|钩子操作|含义|说明|
|----|----|----|----|
|`d.__len__()`|`len(d)`|||
|`d.__getitem__(key)`|`d[key]`|||
|`d.__contains__(key)`|`key [not ]in d`|||
|`d.__iter__()`|`iter(d)`|字典键迭代器||
|`d.get(key[, default])`||带默认值获取元素||
|`d.copy()`||||
|`classmethod fromkey(iterable[, value])`||以可迭代对象为键创建字典||
|`d.__setitem__(key,value)`|`d[key]=value`|||
|`d.__delitem__(key)`|`del d[key]`|||
|`d.clear()`||||
|`d.setdefault(key[, default])`||不存在则设置默认值的键值对|返回值|
|`d.pop(key)`||||
|`d.popitem()`||按 *LIFO* 顺序弹出||
|`d.copy()`||||
|`d.update(other)`|`|=`|在位更新|`|` 即为创建新字典|
|`d.keys()`||字典键视图对象||
|`d.values()`||字典值视图对象||
|`d.items()`||字典键值对视图对象||

####	字典视图对象

-	字典视图对象：提供字典条目的**动态视图**，随字典改变而改变
	-	功能
		-	字典视图可被迭代以产生对应数据
		-	支持成员检测：`len`、`iter`、`in`、`reversed`
		-	若其成员（键、键值对）*hashable*，则视图类似集合，`collection.abc.Set` 定义的操作适用
		-	可通过 `mapping` 属性获取原始字典
	-	创建方式：上述 `dict` 方法

##	`GenericAlias` 类型
#TODO

##	其他类型

-	迭代器类型
	-	需要多次迭代时，应该将迭代器转换为可重复迭代数据结构（迭代器项会被消耗）

-	上下文管理器类型




