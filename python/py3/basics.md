#	Python基础

##	原生数据类型

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

###	注意

-	python是强类型语言，注意区分`"1"`和`1`

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

##	环境系统

###	`os`

-	`chdir`：改变当前目录

####	`environ`

存储python运行时环境变量字典

-	默认继承系统所有环境变量
-	可以通过修改、增加、删除项目在python中动态设置运行时环境
	变量
-	即除`export`环境变量、临时环境变量，python还有第三种
	在运行时动态修改环境变量

##	Python代码技巧

