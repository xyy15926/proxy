#	Python基础

##	原生数据类型

-	python是强类型、动态语言

###	List

####	List的边界、Index问题

-	将list的index视为元素的index而不是分割的index
-	将边界视为左闭右开`[)`的区间（0开始）
-	index可以为负数
	-	但是应当将其对len取模，len+index(<0)
	-	尤切片操作涉及左、右边界，应取模以[0,len-1]中的
		“标准”index作为左（小）右（大）边界

####	相关函数

```python
List = sorted(
	iterable(iterable),
	/,
	*,
	key=None,
	reverse=False)
	# `key`：自定义排序排序函数
		# 以iterable元素为参数
		# 根据返回值排序

List = list1 + list2
	# 列表对象直接相加

None = list1.append(
	[object](val)
)
	# 添加元素至list末尾

None = list1.insert(
	[index](int),
	[object](val)
)
	# 这两个参数都是POSITION_ONLY类型

None = list1.remove(
	[value](val)
)
	# 删除按顺序**第一个**值相同的元素
```

###	Dict

-	dict中的键值对是**无序的**，想要得到有序的值列表，不能
	直接`.values()`，必须提供额外的排序（键list之类）

```python
d = {key: val}
d = dict([(key, val)])
d = dict(key=val)
d = dict.fromkeys(key_list, default_val)
```

####	相关函数

```python
Val = dict1.get(k[, d])
	# 存在则返回`dict1[k]`，否则返回`d`

Val = dict1.setdefault(k[, d])
	# 存在则返回`dict1[k]`，否则设置为`d`并返回

Val = getattr(
	object,
	name,
	[default])
	# 获取`object` `name`属性
	# `default`未设置，`name`不存在则`AttributeError`

(key, val) = dict1.items()

bool = dict.has_key(str)

dict_keys = dict1.keys()

dict_values = dict1.values()

```

###	Tuple

###	Set

####	集合运算

```python
a = set_1 | set_2
set_1.union(set_2)
	# 并集
b = set_1 & set_2
set_1.intersection(set_2)
	# 交集
c = set_1 - set_2
set_1.differenct(set_2)
	# 差集
b = set_1 ^ set_3
set_1.symmetric_difference(set_2)
	# 对称差集

t.add(1)
t.remove(1)
t.update([1, 2, 3])
s.issubset(t)
	# `s`是否是`t`的子集
t.copy()
```

###	Str

####	Py2

python2中`str`类型实际上应该看作是**字节串**

-	`str`索引是按字节索引
-	`decode`方法用于解码，将字节流转换为真正**字符串**

####	Py3

python3中`str`类型的逻辑变成真正的**字符串**

-	python3中`str`类型在内存中应该是`utf-16-le`编码

	-	权衡了内存需求、处理方便
	-	这个和python默认文件、输入、输出编码是两个概念

-	`encode`方法，将字符串编码为其他编码方案的字节流
	（包括自身的*utf-8*）

-	这里的字符串中每个字符内存大小可能不同

###	Bytes

`bytes`类型类似python2中的`str`

-	可以打印字符按*ASCII*码打印，否则打印16进制数字

##	符号

###	运算符

```python
~True
	# `~`取反
	# == -2

1^2
	# `^`异或
	# == 3
```

###	小括号问题
	
-	python中在大部分数据类型（包括list、tuple）外面套上
	小括号表示是该元素
-	`(item,)`才是表示只含一个元素的tuple

##	函数语法

###	函数参数

####	函数签名中参数

#####	POSITIONAL_OR_KEYWORD（常用）

参数前没有*VAR_POSITIONAL*类型的参数，可以通过**位置**或
**关键字**传值

>	注意：POSITION_OR_KEYWORD有默认值参数必须位于无默认值者
	之后，但是KEYWORD_ONLY有默认值参数是可以位于无默认值者
	之前

```python
def say_hello(name):
	print("hello {0}".format(name))

say_hello("jack")
	# 通过位置传值
say_hello(name="tom")
	# 通过关键字传值
```

#####	VAR_POSITIONAL（常用）

即`*args`参数，只能通过**位置**传值

```python
def say_hello(*args):
	print("hello {0}".format(args))

say_hello("jack", "tom")
```

#####	VAR_KEYWORD（常用）

即`**kwargs`参数，只能通过关键字传值

```python
def func_b(**kwargs):
	print(kwargs)

func_b(a=1, b=2)
```

#####	KEYWORD_ONLY

参数前存在*VAR_POSITION*（`*args`）类型的参数，只能通过
**关键字**传值

-	也可以是在参数前加上占位符`*`
-	强制要求keyword传参，意义明确

```python
def func_b(*args, a, b):
	print(args, a, b)

def func_b(*, a, b):
	print(a, b)


func_b("test", a=1, b=2)
```

#####	POSITIONAL_ONLY

只能通过位置传值的参数，Python没有明确的语法定义一个只能通过
位置传值的参数，但在很多内置、扩展模块的函数中接受这种类型的
参数，大概率python中其他语言编写的二进制包，比如numpy中
`np.random.randn`

#####	获取函数参数

需要用到`inspect.signature`获取函数方法签名

```python
import inspect

def func_a(arg_a, *args, arg_b="hello", **kwargs):
	print(arg_a, arg_b, args, kwargs)

if __name__ == "__main__":
	func_signature = inspect.signature(func_a)
		# 获取函数签名
	func_args = [ ]

	for k, v in func_signature.parameters.items():
		# 获取所有参数，并判断参数类型
		if str(v, kind) in ("POSITIONAL_OR_KEYWORD", "KEYWORD_ONLY"):
			# 这两种参数可能有默认值，需要区分
			if isinstance(v.default, type) and v.default.__name__ == "_empty":
				# 若参数无默认值，则`v.default`值为
				# `class inspect_empty`，是`type`类的一个实例
				# 且`v.default.__name__`为`__empty`
				func_args.append({k: None})
			else:
				func_args.append({k: v.default})
		elif str(v.kind) == "VAR_POSITIONAL":
			args_list = [ ]
			func_args.append(args_list)
		elif str(v.kind) == "VAR_KEYWORD":
			args_dict = {}
			func_args.append(args_dict)

	print(func_args)
```

####	函数签名常见模式

```python
def func(essential, optional=val_op, *args, **kwargs):
	pass
```

常用参数类型为KEYWORD_OR_POSITION、POSITION_ONLY、
KEYWORD_ONLY，将KEY_WORD_POSITION根据是否有默认值分为
`essential`、`optional`

-	`essential`：必须参数，没有默认值所以必须传递参数
-	`optional`：缺省参数，有默认值可以不传参
-	`\*args`：不定长参数，参数未命名，传递的参数位于args这个
	tuple里
-	`\*\*kwargs`：关键字参数，参数名称由传参时命名

注意：

-	定义函数时需要要按照以上顺序定义参数，参数类型和所处位置
	有关（KEYWORD_ONLY和POSITION_OR_KEYWORD）
-	args、kwargs参数中也可能有“必须”参数，缺少时，虽然
	调用不会报错，但是函数内部会出现“越界”错误
-	参数默认值应该使用值无法更改的数据类型，否则可能在执行
	过程中发生变化

####	调用原则

-	可选传参放在必须传参之后
-	关键字传参放在无关键字传参之后
-	无关键字传参按顺序
-	关键字传参可无序

发现一个有趣的问题

-	定义全部四种参数
-	但是函数里面不使用`kwargs`，使用`**kwargs`
-	这时函数可以执行，但是不能使用关键字参数，且`**kwargs`
	为空

####	传参

python只有一种参数传递方式：传对象引用

-	python中一切都是对象
-	list、dict这类可更改对象，原值可被更改，相当于传地址
-	int、tuple之类的不可更改对象，原值不可更改，相当于传值

其实可以看作是传“地址（引用）”

-	不可更改对象，引用传递后值无法更改，只能创建新对象，而
	不会影响原对象
-	可更改对象，引用传递后可以直接更改，影响到原对象

###	变量作用域

-	python中，对变量的**搜索**按照LEGB规则进行，“内层”
	作用域可以**访问（修改无效）**“外层”作用域变量
	-	local：本地作用域
	-	enclosing function locals：嵌套作用域
	-	global：全局作用域
	-	builtins：内置变量作用域

-	但在函数内部为变量**赋值**时，并不是按照LEGB规则搜索再
	赋值，而总是**创建或改变local的变量名（对应变量），除非
	在函数中已经被声明为全局变量**

####	`global`

`global`关键字**获取、创建**global作用域中的变量，然后可以
对其进行修改

-	`global`修饰的变量之前可以不存在，即可以创建全局变量

```python
x = 99

def local_change():
	x = 88

def global_func():
	global x
	x = 88

local_change()
print(x)
	# `x`未改变，输出99

global_func()
print(x)
	# `x`发生改变，输出88
```

####	`nonlocal`

`nonlocal`可以在被嵌套函数中**获取**嵌套作用域中的变量，然后
可以对其修改

-	`nonlocal`修饰（获取）的变量必须在嵌套作用域中已存在

```python
def func():
	count = 1
	def change():
		count = 12
	def outer_change():
		count 12

	change()
	print(count)
		# `count`未改变，仍然输出1
	outer_change()
	print(12)
		# `count`改变，输出12
```

##	函数说明

###	List类型

####	`zip(iterable,...)`

-	参数：一个或多个iterable类型

-	返回值

	-	将iterable参数的各对位元素打包成tuple，返回由这些
		tuple组成的zip对象（iterable，python2中好像是直接
		返回list）

	-	参数长度不一致，以最短为准

```python
zip(list[:-1], list[1:])
	# 返回list前后相接的tuple
zip(*zip_obj)
	# 返回得到zip_obj的iterable参数组成的zip对象（但为tuple）
```

###	输入输出

> - 内建`print`、`input`函数实际上只是调用`sys.stdin`、
	`sys.stdout`引用文件流对象的`readline`、`write`方法

####	`print`

```python
def print(value,...,
	sep=' '/str,
	end='\n'/str,
	file=sys.stdout/file,
	flush=False/True
)
def input(prompt=None,/)
```

-	参数
	-	`sep`：插入值之间标记字符串
	-	`end`：添加在最后的标记
	-	`file`：文件类型对象
	-	`flush`：是否强制刷新流

####	`input`

```python
def input(prompt)
```

-	`input`方法会自动剔除行结尾的`\n`，遇到*EOF*会抛出异常
	-	标准输入重定向为文件时
	-	键盘输入`<c-d>`/`<c-z-cr>`

###	文件处理

```python
file = open("data.txt", "w")
	// 打开文件，创建流
file.write("hello world\n")
	// 逐字符写入字符串
	// 需要手动添加`\n`
file.read([N])
	// 缺省读取全部
file.readline()
	// 读取一行
file.readlines()
	// 读取全部，返回行字符串列表
file.seek(0)
file.flush()
	// 强制缓冲区数据写入磁盘
file.close()
	// 确定文件内容、释放系统资源
```

####	`open`

```python
def open(file,
	mode="r"/"w"/"a"/"+"/"t"/"b",
	buffering=-1,
	encoding=None,
	errors=None,
	newline=None,
	closefd=True,
	opener=None
)
```

####	`.close`

`.close`调用经常是可选的

-	垃圾回收、进程终止时会关闭文件

-	但某些情况下需要显式关闭

	-	Jython实现依赖于Java的垃圾回收，无法像在标准python
		中那样确定文件回收时间
	-	短时间内新建许多文件
	-	IDE对文件持有时间过长，甚至可能导致缓冲区数据未存盘
	-	在同一进程中再次打开文件

-	自动关闭文件特性是实现特性，不是语言特性，将来可能会
	改变

> - 关闭文件没有坏处，还能形成好习惯

#####	关闭文件处理方式

```python
f = open("filename")
try:
	...
finally:
	f.close()
	# 异常处理器

with open("filename") as f：
	...
	# 上下文管理器
```

-	文件对象的自动关闭、收集性通常能满足要求，没有必要把所有
	python文件处理代码封装在`with`中

####	`seek`

```python
file.seek(offset, start)
	// start: 0: 文件开始，1：当前位置，2：文件结尾
```

-	可以在非更新模型（不带上`+`）下使用，但是在更新模式下
	最为灵活

-	除非是纯ASCII文件，否则由于编码问题、换行符问题的存在，
	其在文本应用中效果不好，所以经常在二进制文件+二进制模式
	使用

###	解释器

```python
id(obj)
	# 返回`object identifier
locals()
	# 返回当前环境所有定义值字典
```

##	环境系统

###	编码

> - 参见*cs_program/character/char_encoding*

####	Py2

-	默认编码方案是*ASCII*（str类型编码方案）

-	接受*ASCII*字符集以外的字符时，也认为是*ASCII*编码字节流
	，并采用码元长度`1B`存储

-	和*ASCII*编码方案不兼容的其他编码方案存储的文件会因为
	python2无法“理解”报错

####	Py3

-	默认编码方案为*utf-8*
-	默认将输入字节流视为*utf-8*编码

###	*CWD*

*CWD*：当前工作目录，可以使用`os.getcwd()`获得

-	默认是启动python进程的目录，除非使用`os.chdir()`修改
-	没有提供完整路径的文件名将被映射到*CWD*
-	与模块导入路径无关


