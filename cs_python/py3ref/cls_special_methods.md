---
title: Special Methods
categories:
  - Python
  - Py3Ref
tags:
  - Python
  - Py3Ref
  - Routine
  - Method
date: 2019-06-04 00:55:56
updated: 2019-06-04 00:55:56
toc: true
mathjax: true
comments: true
description: Special Methods
---

##	综述

特殊方法：python类中具有特殊名称的方法，实现**由特殊语法**
所引发的特定操作

-	python实现**操作符重载**的方式
	-	允许每个类自行定义基于操作符的特定行为
	-	特定操作包括python内置的**钩子函数**

-	钩子函数不能简单的看作**直接调用**特殊方法
	-	尝试调用备用实现：`iter`、`reversed`
	-	修改方法返回值：`dir`

-	大部分情况下，若没有定义适当方法，尝试**执行操作**将
	`raise AttributeError`、`raise TypeError`

	-	但`__hash__`、`__iter__`、`__reversed__`、
		`__contains__`等方法即使未定义，其**对应钩子函数**
		实现会尝试调用可能的其他方法完成操作
		（直接`obj.__xxx__`调用方法仍然报错）

	-	将特殊方法设为`None`表示对应操作不可用，此时即使以上
		`hash`、`iter`、`reversed`、`in`等操作也不会尝试调用
		备用方法

##	实例创建、销毁

调用类时，元属性方法执行顺序

-	`__prepare__()`：创建命名空间
-	依次执行类定义语句
-	`__new__()`：创建类实例
-	`__init__()`：初始化类
	-	`__new__`**返回的新实例**的`__init__`方法将被调用
	-	用户定义`__new__`返回对象不一定期望类实例，调用的
		`__init__`随之不一定是期望方法
-	返回`__new__`返回类实例

###	`__prepare__`

-	在所有类定义开始执行前被调用，用于创建类命名空间
-	一般这个方法只是简单的返回一个字典或其他映射对象

###	`__new__`

```python
classmethod object.__new__(cls[, *args, **kwargs]):
	pass
```

-	用途：创建、返回`cls`类新实例
	-	`super().__new__(cls[,...])`调用超类方法创建类实例，
		然后根据需要修改新创建实例再返回

-	参数
	-	`cls`：待实例化类
	-	其余参数：类构造器表达式参数

-	返回值：`cls`类新实例
	-	`__new__`返回值就是类构造器的返回值，有绝对控制权

####	说明

-	`__new__`：*builtin_function_or_method*

-	`__new__`是静态方法：以需实例化类作为第一个参数
	-	`__new__`方法绑定当前**类对象**
	-	特例，不需要显式声明为静态方法

-	原生有两个`__new__`函数，二者C实现不同
	-	`type.__new__`：元类继承，用于创建类对象
	-	`object.__new__`：其他类继承，用于创建实例

###	`__init__`

```python
def object.__init__(self[, *args, *kwargs]):
	pass
```

-	用途：初始化类实例
	-	类构造器中`__new__`返回类实例调用此方法初始化
	-	若基类有用户定义`__init__`方法，则其派生类`__init__`
		应该**显式调用**基类`__init__`保证基类部分正确初始化

-	参数
	-	`self`：当前类实例
	-	其余参数：类构造器表达式参数

-	返回值：`None`，否则`raise TypeError`

###	`__del__`

```python
def object.__del__(self)
```

-	用途：实例销毁时（引用计数变为0）被调用
	-	若基类有`__del__`方法，则其派生类`__del__`方法中
		需要**显式调用**基类`__del__`保证基类部分正确清除
	-	对象重生：在其中创建该实例的新引用推迟其销毁
		-	不推荐
		-	重生对象被销毁时`__del__`是否会被再次调用取决于
			具体实现
		-	当前CPython实现中只会调用一次

####	说明

-	解释器退出时不会确保为仍然存在的对象调用`__del__`方法
-	“钩子函数”：`del`
	-	`del x`不直接调用`x.__del__()`
	-	`del x`仅将`x`的引用计数减一

##	输出属性

###	`__repr__`

```python
def object.__repr__(self):
	pass
```

-	用途：输出对象的“官方”字符串表示
	-	如果可能，应类似有效的python表达式，可以用于重建具有
		相同取值的对象（适当环境下）
	-	若不可能，返回形如`<...some useful description...>`
		的字符串
	-	常用于调试，确保内容丰富、信息无歧义很重要

-	返回值：字符对象
	-	内置钩子函数：`repr`
	-	**交互环境**下直接“执行”变量的结果

###	`__str__`

```python
def object.__str__(self):
	pass
```

-	用途：生成对象“非正式”、格式良好的字符串表示
	-	返回较方便、准确的描述信息

-	返回值：字符串对象
	-	内置钩子函数：`str`

####	说明

-	`object.__str__`方法默认实现调用`object.__repr__`
	-	所以若未定义`__str__`，需要实例“非正式”字符串表示时
		也会使用`__repr__`

-	`format`、`print`函数会隐式调用对象`__str__`方法
	-	此时若`__str__`返回非字符串会`raise TypeError`

###	`__bytes__`

```python
def object.__bytes__(self):
	pass
```

-	用途：生成对象的字节串表示

-	返回值：`bytes`对象
	-	内置钩子函数：`bytes`

###	`__format__`

```python
def object.__format__(self, format_spec)
```

-	用途：生成对象的“格式化”字符串表示
	-	内部常调用`format`、`str.format`实现格式化
	-	`object.__format__(x, '')`等同于`str(x)`

-	参数
	-	`fomrat_spec`：包含所需格式选项描述的字符串
		-	参数解读由实现`__format__`的类型决定
		-	大多数类将格式化委托给内置类型、或使用相似格式化
			语法

-	返回值：字符串对象
	-	内置钩子函数：`format`

###	`__hash__`

```python
def object.__hash__(self):
	pass
```

-	用途：计算对象hash值返回
	-	相等的对象（即使类型不同）**理应**具有相同hash值
	-	建议把参与比较的对象的全部组件的hash值打包为元组，
		对元组做hash运算
		```python
		def __hash__(self):
			return hash((self.name, self.nick, self.color))
		```

-	返回值：整数
	-	内置钩子函数：`hash()`

####	说明

-	`hash()`会从对象自定义的`__hash__()`方法返回值中截断为
	`Py_ssize_t`大小
	-	64bits编译平台通常为8bytes、32bits为4bytes
	-	若对象`__hash__()`需要在不同位大小的平台上互操作，
		需要检查支持的平台位宽

	> - 查看`sys.hash_info.width`

-	`set`、`frozenset`、`dict`这3个hash集类型中成员的操作
	会调用相应`__hash__()`

-	类的`__hash__`方法设置为`None`时
	-	尝试获取实例hash值时将`raise TypeError`
	-	`isinstance(obj, collecitons.abc.Hashable)`返回
		`False`
	-	单纯在`__hash__`中显式`raise TypeError`会被错误
		认为是可hash

####	关联`__eq__`

hash绝大部分应用场景是比较是否相等，所以`__hash__`、`__eq__`
密切相关

-	类未定义`__eq__`
	-	也不应该定义`__hash__`，单独hash结果无法保证比较结果

-	类实现`__eq__`
	-	未定义`__hash__`：其实例将不可被用作hash集类型的项
	-	类中定义了可变对象：不应该实现`__hash__`，因为hash集
		实现要求键hash值不可变

-	类重载`__eq__`方法
	-	默认其`__hash__`被隐式设为`None`
	-	否则须设置`__has__ = <ParentClass>.__hash__`显式保留
		来自父类`__hash__`实现

####	默认实现

-	`float`、`integer`、`decimal.Decimal`等数字类型hash运算
	是基于为任意有理数定义的统一数学函数

	> - 详细参考<https://docs.python.org/zh-cn/3/library/stdtypes.html#hashing-of-numeric-types>

-	`str`、`bytes`、`datetime`对象`__hash__`值会使用不可预知
	值**随机加盐**
	-	盐在单独python进程中保持不变，但在重复执行的python
		进程之间是不可预测的
	-	目的是为了防止某种形式的DDos服务攻击

-	改变hash值会影响集合迭代次序
	-	python也不保证次序不会改变

###	`__bool__`

```python
def object.__bool__(self):
	pass
```

-	用途：返回`True`、`False`实现真值检测
	-	未定义：调用`__len__`返回非0值时对象逻辑为真
	-	`__len__`、`__bool__`均未定义：所有实例逻辑为真

-	返回值：`False`、`True`
	-	内置构造函数：`bool()`

###	例

```python
class Pair:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __repr__(self):
		# 返回实例代码表示形式
		# 通常用于重新构造实例
		return "Pair({0.x!r}, {0.y!r})".format(self)
			# 格式化代码`!r`指明输出使用`__repr__`而不是默认
				# 的`__str___`
			# 格式化代码`0.x`表示第一个参数`x`属性

	def __str__(self):
		return "({0.x!s}, {0.y!s})".format(self)
			# 格式化代码`!s`指明使用默认`__str__`

	def __format__(self):
		if self.x == 0:
			return self.y
		elif self.y == 0:
			return self.x
		return "{0.x!r}, {0.y!r}".format(self)
```

##	*Rich Comparison Methods*

富比较方法

```python
def object.__lt__(self, other):
	pass
def object.__le__(self, other):
	pass
def object.__eq__(self, other):
	pass
def object.__ne__(self, other):
	pass
def object.__gt__(self, other):
	pass
def object.__ge__(self, other):
	pass
```

-	用途：比较运算符重载
	-	`x < y`：调用`x.__lt__(y)`
	-	`x <= y`：调用`x.__le__(y)`
	-	`x == y`：调用`x.__eq__(y)`
	-	`x != y`：调用`x.__ne__(y)`

-	返回值
	-	成功比较返回`False`、`True`
	-	若指定方法没有相应实现，富比较方法会返回单例对象
		`NotImplemented`

> - 比较运算默认实现参见*cs_python/py3ref/expressions*

###	说明

-	默认情况下，`__ne__`会委托给`__eq__`，并将结果取反，除非
	结果为`NotImplemented`

-	比较运算符之间没有其他隐含关系
	-	`x < y or x == y`为真不意味着`x <= y`
	-	要根据单个运算自动生成排序操作可以利用
		`functools.total_ordering()`装饰器简化实现

-	以上方法没有对调参数版本（左边参数不支持该操作，右边参数
	支持该操作）
	-	若两个操作数类型不同、且右操作数是左操作数直接或间接
		子类，优先选择右操作数的**反射方法**，否则左操作数
		方法（不考虑虚拟子类）
	-	反射方法
		-	`__lt__`、`__gt__`互为反射
		-	`__le__`、`__ge__`互为反射
		-	`__eq__`、`__ne__`各为自身反射

##	内部信息

###	`__dict__`

-	钩子函数：`vars`、`dir`（部分）

	-	`vars`是真正对应的钩子函数，返回键值对
	-	`dir`执行过程中会访问`__dict__`、`__class__`，而且
		只返回keys

-	**对象**底层字典，存储对象属性、方法

	-	注意区分开：实例属性、类属性、基类属性，`__dict__`
		只包括**当前实例**属性、方法
	-	返回结果是`dir`结果的子集

-	调用实例`obj`的属性时，按照以下顺序查找

	-	`obj.__dict__`：当前实例的`__dict__`中
	-	`type(obj).__dict__`：实例所属类的`__dict__`中
	-	`type(obj).mro().__dict__`：基类的`__dict__`中

-	在大部分情况下`__dict__`会自动更新，如`setattr`函数时，
	或说实例的属性、方法更新就是`__dict__`的变动

	-	一般情况下不要直接访问`__dict__`，除非真的清楚所有
		细节，如果类使用了`cls.__slots__`、`@property`、
		描述器类等高级技术时代码可能会被破坏

	-	尽量使用`setattr`函数，让python控制其更新

###	`__class__`

-	用途：返回实例所属类

-	返回值：实例（狭义）返回类、类返回元类
	-	钩子函数：`type`

###	`__objclass__`

-	用途：被`inspect`模块解读为**指定实例所在的类**
	-	合适的设置可以有助于动态类属性的运行时检查
	-	对于可调用对象：指明第一个位置参数应为特定类型的
		实例、子类
		-	描述器类：`instance`参数
	#todo

###	`__slots__`

-	用途：显式声明数据成员、特征属性，限制实例添加属性

	-	可赋值为：字符串、可迭代对象、实例使用的变量名构成的
		字符串序列

		-	可迭代对象中元素可以是任何类型
		-	还可以映射类型，未来可能会分别赋给每个键特殊含义
			的值

	-	`__slots__`会为已声明变量保留空间
		-	直接访问将`raise AttributeError`
		-	`dir`可以找到`__slots__`中声明的变量

-	阻止默认为每个实例创建`__dict__`、`__weakref__`的
	行为，除非在`__slots__`中显式声明、或在父类中可用

	-	无`__dict__`属性实例无法给未在`__slots__`中列出
		的新变量赋值

		-	但是python很多特性依赖于普通的依赖字典实现，定义
			`__slots__`的类不再支持普通类某些特性
		-	大多数情况下，应该只在经常使用到作为数据结构的
			类上定义`__slots__`
		-	不应该把`__slots__`作为防止用户给实例增加新属性
			的封装工具

	-	无`__weakref__`属性实例不支持对实例的弱引用

> - 是阻止给实例创建`__dict__`，类本身仍然有`__dict__`属性
	（`dir`返回值中无`__dict__`，`__dir__`返回值中有）

####	说明

-	`__slots__`声明的行为不只限于定义其的类

	-	父类中声明`__slots__`可以在子类中使用，但子类将获得
		`__dict__`、`__weakref__`，除非其也定义了`__slots__`

	-	子类`__slots__`中定义的slot将覆盖父类中同名slot
		-	需要直接从基类直接获取描述器才能访问
		-	这会导致程序未定义，以后增加检查避免

	-	多重继承中只允许一个父类具有非空`__slots__`，否则
		`raise TypeError`

-	`__slots__`是在类层次上的实现：为每个变量创建描述器

	-	类属性不能被用于给在`__slots__`中定义变量设置默认值
	-	否则类属性会覆盖描述器赋值，变成只读属性

-	非空的`__slots__`不适用于派生自“可变长度”内置类型，如
	`int`、`bytes`、`tuple`

-	定义类属性`__slots__`后，python会为实例属性使用紧凑内部
	表示
	-	实例属性使用固定大小、很小的数组构建，而不是为每个
		实例定义字典
	-	在`__slots__`列出的属性名在内部映射到数组指定下标上
	-	类似于R中`factor`类型、C中`enum`类型
	-	相比`__dict__`可以显著节省空间、提升属性查找速度

```python
class Date:
	__slots__ = ["year", "month", "day"]
	def __init__(self, year, month, day):
		self.year = year
		self.month = month
		self.day = day
```


-	继承自未定义`__slots__`类时，实例中`__dict__`、
	`__weakref__`属性将总是可访问的

-	`__class__`赋值仅在两个类具有相同`__slots__`值时才有用

##	自定义属性访问

###	`__getattr__`

```python
def object.__getattr__(self, name):
	pass
```

-	用途：`.`默认属性访问引发`AttributeError`而失败时调用
	-	如果属性通过正常机制找到，`__getattr__`不会被调用
		-	在`__getattr__`、`__setattr__`之间故意设置的
			不对称性
		-	出于效率考虑
	-	对实例变量而言，无需在实例属性字典中插入值，就可以
		模拟对其的完全控制

-	返回值：计算后的属性值、或`raise AttributeError`

####	说明

-	可能引发`AttributeError`
	-	调用`__getattribute__`时因为`name`不是实例属性、
		或是类关系树中属性
	-	对调用`__get__`获取`name`描述器

-	调用`__getattr__`是`.`运算符中逻辑
	-	`__getattribute__`显式调用`raise AtttributeError`
		不会调用`__getattr__`

-	`__getattr__`甚至不是`object`具有的
	`<wrapper_descriptor>`

-	相较于`__getattribute__`其实更常用，因为修改**所有**对
	对对象的访问逻辑没啥价值

###	`__getattribute__`

```python
def __getattribute__(self, key):
	"Emulate type_getattro() in Objects/typeobject.c"
	v = object.__getattribute__(self, key)
	if hasattr(v, "__get__"):
		return v.__get__(None, self)
	return v
```

-	用途：访问对象属性时无条件被调用
	-	**判断访问属性类型、做对应操作**
		-	描述器：调用描述器方法
		-	实例方法：为类中函数绑定实例
		-	类方法：为类中函数绑定类
		-	静态方法：不绑定
		-	普通属性
	-	作为**通过特定语法、内置函数隐式调用的结果**情况下，
		查找特殊方法时仍可能被跳过

-	返回值：找到的属性值、或`raise AttributeError`

> - `__getattribute__`仅对继承自`object`的新式类实例可用

####	说明

-	内置类型均有各自`__getattribute__`函数实例
	-	其均为`wrapper_descriptor`类型（C实现的函数）
	-	各函数实例标识符不同，若其均“继承自`object`”，其
		应为同一个函数实例
	-	自定义类真继承自`object`类，其`__getattribute__`同
		`object.__getattribute__`

-	自定义实现
	-	为**避免方法中无限递归**，实现总应该调用具有相同名称
		基类方法访问所需要的属性

####	钩子函数

-	`.`运算符：首先调用`__getattribute__`，若无访问结果，
	调用`__getattr__`

	> - `.`运算符说明参见*cs_python/py3ref/cls_basics*

-	`getattr`：基本同`.`运算符，除可捕获异常，设置默认返回值

-	`hasattr`：内部调用`getattr`，根据`raise Exception`判断
	属性是否存在

	-	可以通过`@property.getter`中`raise AttributeError`
		使得属性看起来不存在
	-	内部有更多`boilerplate`相较于`getattr`更慢
	-	则按照字面意思使用不需要考虑过多

###	`__setattr__`

```python
def object.__setattr__(self, name, value):
	pass
```

-	用途：**属性被尝试赋值时被调用**
	-	默认实现：将值保存到实例字典
	-	若`__setattr__`要赋值给实例属性，应该调用同名基类
		方法

-	返回指：`None`
	-	钩子函数：`setattr`

###	`__delattr__`

```python
def object.__delattr__(self, name):
	pass
```

-	用途：**删除实例属性时被调用**
	-	默认实现：从实例字典中删除对应项
	-	应该在`del obj.name`对该对象有意义时才实现

-	返回值：`None`
	-	内置钩子函数：`delattr`、`del`

###	`__dir__`

```python
def object.__dir__(self):
	pass
```

-	用途：返回实例中“可访问”名称的字符串列表
	-	默认实现：返回实例、类、祖先类所有属性
	-	交互式解释器就是在`__dir__`/`dir`返回列表中进行查询
		进行补全

-	返回值：序列
	-	内置钩子函数：`dir`
		-	`dir()`获取`__dir__`返回序列，转换为列表、排序
		-	`dir()`会剔除`__dir__`返回值中部分值
		-	若`__dir__`返回值不可迭代，报错

###	自定义模块属性访问

-	`__getattr__`、`__dir__`可以用于自定义对模块属性的访问
	-	模块层次`__getattr__`类似普通类
		-	接受属性名作为参数
		-	返回计算后结果、或`raise AttributeError`
		-	若正常查找`__getattribute__`无法在模块中找到某个
			属性，调用`__getattr__`
	-	模块层次`__dir__`类似普通类
		-	不接受参数
		-	返回模块中可访问名称的字符串列表

-	可以将模块的`__class__`属性设置为`types.ModuleType`子类

	```python
	import sys
	import types import ModuleType

	class VersboseModule(ModuleType):
		def __repr__(self):
			return f"verbose {self.__name__}"
		def __setattr__(self, attr, value):
			print(f"settting {attr}")
			super().__setattr__(attr, value)

	sys.modules[__name__].__class__ = VerboseModule
	```

> - 设置模块`__getattr__`、`__class__`只影响使用**属性访问**
	语法进行查找，直接访问模块全局变量（通过模块内代码、对
	模块全局字典引用）不受影响

##	描述器类

描述器：**具有“绑定行为”**的对象属性

-	类中定义其中任意一个方法，则其实例被称为描述器
	-	`__set__`
	-	`__get__`
	-	`__delete__`

-	所有对描述器属性的访问会被`__get__`、`__set__`、
	`__delete__`方法捕获/重载

	-	如果只是想简单的自定义某个类的属性处理逻辑，使用
		`@porperty`装饰器简化实现

> - `@property`参见*cs_python/py3ref/cls_basics*

###	描述器协议

-	以下方法仅包含其的类的实例出现在类属性中才有效
	-	即以下方法必须在（祖先）类`__dict__`中出现，而不是
		实例`__dict__`中
	-	即描述器只能定义为类属性，不能定义为实例属性

####	`__get__`

```python
def object.__get__(self, instance, owner=None):
	pass
```

-	用途：访问描述器属性时调用，重载实例属性访问
	-	若描述器未定义`__get__`，则访问属性会返回描述器对象
		自身，除非实例字典`__dict__`中有同名属性
	-	若仅仅只是从底层实例字典中获取属性值，`__get__`方法
		不用实现

-	参数
	-	`instance`：用于方法属性的实例
	-	`owner`：实例所属类，若通过类获取属性则为`None`

-	返回值：计算后属性值、或`raise AttributeError`

-	示例

	```python
	def __get__(self, instance, cls):
		if instance is None:
			# 装饰器类一般作为类属性，需要考虑通过类直接访问
				# 描述器类属性，此时`instance is None`
			# 常用操作是返回当前实例
			return self
		else:
			return instance.__dict__[self.name]

		# self：描述器类当前实例
		# instance：定义描述器作为类属性的类的实例
		# cls：定义描述器作为类属性的类
	```

####	`__set__`

```python
def object.__set__(self, instance, name, value):
	pass
```

-	用途：设置实例`instance`的“描述器属性”值为`value`，重载
	实例属性赋值
	-	常用实现：操作实例`instance.__dict__`存储值，使得
		看起来是设置普通实例属性

-	示例

	```python
	def __set__(self, instance, name, value):
		if instance is None:
			pass
		else:
			if not instance(value, int):
				raise TypeError("expect an int")
			instance.__dict__[self.name] = value
			# 操作实例底层`__dict__`

		# `value`：赋给描述器类属性的值
	```

####	`__delete__`

```python
def object.__delete__(self, instance):
	pass
```

-	用于：“删除”实例`instance`的“描述器属性”，重载实例属性
	删除
	-	具体实现应取决于`__set__`实现

-	示例

	```python
	def __delete__(self, instance):
		if instance is None:
			pass
		else:
			del instance.__dict__[self.name]
			# 操作实例底层`__dict__`
	```

####	`__set_name__`

```python
def object.__set_name__(self, owner, name):
	pass
```

-	用途：类`owner`被创建时调用，描述器被赋给`name`

###	实现原理

-	描述器的实现依赖于`object.__getattribute__()`方法
	-	 可以通过重写类的`__getattribute__`方法改变、关闭
		描述器行为

-	描述器调用：描述器`x`定义在类`A`中、`a = A()`
	-	直接调用：`x.__get__(a)`
	-	实例绑定：`a.x`
		-	转换为：`type(a).__dict__['x'].__get__(a)`
	-	类绑定：`A.x`
		-	转换为：`A.__dict__['x'].__get__(None,A)`
	-	超绑定：`super(a, A).x`
		-	

####	实例绑定--资料描述器

> - 资料描述器：定义了`__set__`、`__delete__`方法
> - 非资料描述器：只定义了`__get__`方法

-	访问对象属性时，描述器调用的**优先级**取决于描述器定义的方法
	-	优先级：资料描述器 > 实例字典属性 > 非资料描述器
	-	实例属性会重载非资料描述器
	-	实例属性和资料描述器同名时，优先访问描述器，否则优先
		访问属性

-	只读资料描述器：`__set__`中`raise AttributeError`得到

####	描述器调用
#todo

####	Python设计

-	`function`类中定义有`__get__`方法，则其实例（即函数）
	都为非资料描述器
	-	所以实例可以覆盖、重载方法
	-	`__getattribute__`会根据不同方法类型选择绑定对象
		-	`staticmethod`：静态方法
		-	`classmethod`：类方法
		-	实例方法

-	`super`类中定义有`__get__`方法，则其实例也为描述器

-	`@property`方法被实现为资料描述器

###	特殊描述器类
#todo

-	`wrapper_descripter`：`<slot wrapper>`，封装C实现的函数
	-	等价于CPython3中函数
	-	调用`__get__`绑定后得到`<method-wrapper>`
	-	`object`的方法全是`<slot wrapper>`

-	`method-wrapper`：`<method-wrapper>`，封装C实现的绑定方法
	-	等价于CPython3中绑定方法

####	`function`描述器类

`function`描述器类：实例化即得到函数

```python
class function:
	function(code, globals[, name[, argdefs[, closure]]])

	def __call__(self, /, *args, **kwargs):
		# 作为一个函数调用自身

	def __get__(self, instance, owner, /):
		# 返回`owner`类型实例`instance`的属性
		# 即返回绑定方法
```

####	`method`描述器类

`method`描述器类：实例化即得到*(bound )method*，绑定方法

```python
class method:
	method(function, instance)

	def __call__(self, /, *args, **kwargs):
		# 作为函数调用自身

	def __get__(self, instance, owner, /):
		# 返回自身
```

> - *(bound )method*：绑定方法，（首个参数）绑定为具体实例
	的函数，即实例属性

####	`XXmethod`描述类

> - 代码是C实现，这里是python模拟，和`help`结果不同

```python
class classmethod:
	def __init__(self, method):
		self.method = method
	def __get__(self, obj, cls):
		return lambda *args, **kw: self.method(cls,*args,**kw)

class staticmethod:
	def __init__(self, callable):
		self.f = callable
	def __get__(self, obj, cls=None):
		return self.f
	@property
	def __func__(self):
		return self.f
```

-	类中静态方法、类方法就是以上类型的描述器
	-	静态方法：不自动传入第一个参数
	-	类方法：默认传递类作为第一个参数
	-	描述器用途就是避免默认传入实例为第一个参数的行为

-	静态方法、类方法均是非资料描述器，所以和实例属性重名时
	会被覆盖

-	所以类静态方法、类方法不能直接通过`__dict__`获取、调用，
	需要调用`__get__`方法返回绑定方法才能调用

	-	直接访问属性则由`__getattribute__`方法代劳

###	例

```python
class Integer:
	# 描述器类
	def __init__(self, name):
		self.name = name

	def __get__(self, instance, cls):
		# 描述器的每个方法会接受一个操作实例`instance`
		if instance is None:
			# 描述器只能定义为类属性，在这里处理直接使用类
				# 访问描述器类的逻辑
			return self
		else:
			return instance.__dict__(self.name)

	def __set__(self, instance, value):
		if not instance(value, int):
			rasie TypeError("expect an int")
		instance.__dict__[self.name] = value
			# 描述器方法会操作实例底层`__dict__`属性

	def __delete__(self, instance):
		del instance.__dict__[self.name]

class Point:
	x = Integer("x")
	y = Integer("y")
		# 需要将描述器的实例作为类属性放在类的定义中使用

	def __init__(self, x, y):
		self.x = x
		self.y = y

def test():
	p = Point(2, 3)
	print(p.x)
		# 调用`Point.x.__get__(p, Point)`
	print(Point.x)
		# 调用`Point.x.__get__(None, Point)`
	p.y = 5
		# 调用`Point.y.__set__(p, 5)`
```

##	自定义类创建

###	`__init_subclass__`

```python
classmethod object.__init_subclass__(cls):
	pass
```

-	用途：派生类继承父类时，基类的`__init_subclas__`被调用
	-	可以用于编写能够改变子类行为的类
	-	类似类装饰器，但是类装饰其影响其应用的类，而
		`__init_subclass__`影响基类所有派生子类
	-	默认实现：无行为、只有一个参数`cls`
	-	方法默认、隐式为类方法，不需要`classmethod`封装


-	参数
	-	`cls`：指向新的子类
	-	默认实现无参数，可以覆盖为自定义参数

	```python
	class Philosopher:
		def __init_subclass__(self, default_name, **kwargs):
			super().__init_subclass__(**kwrags)
			cls.default_name = default_name

	class AstraliaPhilosopher(Philosopher, default_name="Bruce"):
		pass
	```

	-	定义派生类时需要注意传递参数
	-	元类参数`metaclass`会被其他类型机制消耗，不会被传递
		给`__init_subclass__`

###	元类

-	默认情况下，类使用`type`构建
	-	类体在新的命名空间中执行，类名被局部绑定到
		元类创建结果`type(name, bases, namespace)`

-	可在类定义部分传递`metaclass`关键字参数，自定义类创建
	过程
	-	类继承同样继承父类元类参数
	-	其他类定义过程中的其他关键字参数会在以下元类操作中
		进行传递
		-	解析MRO条目
		-	确定适当元类
		-	准备类命名空间`__prepare__`
		-	执行类主体
		-	创建类对象

####	解释MRO条目

```c
def type.__mro_entries__():
	pass
```

-	用途：若类定义中基类不是`type`的实例，则使用此方法对
	基类进行搜索
	-	找到结果时，以原始基类元组作为参数进行调用

-	返回值：类的元组替代基类被使用
	-	元组可以为空，此时原始基类将被忽略

####	元类确定

-	若没有显式给出基类、或元类，使用`type()`
-	若显式给出的元类不是`type()`的实例，直接用其作为元类
-	若显式给出`type()`实例作为元类、或定义有基类，则选取
	“最派生”元类
	-	最派生元类从显式指定的元类、基类中元类中选取
	-	最派生元类应为所有候选元类的子类型
	-	若没有满足条件的候选元类则`raise TypeError`

####	准备类命名空间

```python
def type.__prepare__(name, bases, **kwds):
	pass
```

-	用途：确定合适的元类之后，准备类命名空间
	-	若元类没有`__prepare__`属性，类命名空间将被初始化为
		空`ordered mapping`

-	参数：来自于类定义中的关键字参数

####	执行类定义主体

```python
exec(body, globals(), namespace)
	# 执行类主体类似于
```

-	普通调用和`exec()`区别
	-	类定义在函数内部时
		-	词法作用域允许类主体、方法引用来自当前、外部
			作用域名称
		-	但内部方法仍然无法看到在类作用域层次上名称
		-	类变量必须通过实例的第一个形参、类方法方法

####	创建类对象

```python
metaclass(name, base, namespace, **kwds):
	pass
```

-	用途：执行类主体填充类命名空间后，将通过调用
	`metaclass(name, base, namespace, **kwds)`创建类对象

-	参数：来自类定义中的关键字参数

#####	说明

-	若类主体中有方法中引用`__class__`、`super`，则`__class__`
	将被编译器创建为隐式闭包引用

	-	这使得无参数调用`super`可以能基于词法作用域正确
		定位类
	-	而被用于进行当前调用的类、实例则是基于传递给方法
		的第一个参数来标识

##	自定义实例、子类检查

-	以下方法应该的定义在元类中，不能在类中定义为类方法
	-	类似于实例从类中查找方法

-	元类`abc.ABCMeta`实现了以下方法以便允许将抽象基类`ABC`
	作为“虚拟基类”添加到任何类、类型（包括内置类型）中

###	`__instancecheck__`

```python
def class.__instancecheck__(self, instance):
	pass
```

-	用途：若`instance`被视为`class`直接、间接实例则返回真值
	-	重载`instance`内置函数行为

-	返回：布尔值
	-	内置钩子函数：`isintance(instance, class)`

###	`__subclasscheck__`

```python
class.__subclasscheck__(self, subclass):
	pass
```

-	用途：若`subclass`被视为`class`的直接、间解子类则返回
	真值
	-	重载`issubclass`内置函数行为

-	返回：布尔值
	-	内置钩子函数：`issubclass(subclass, class)`

##	模拟范型类型

###	`__class_getitem__`

```python
classmethod object.__class_getitem__(cls, key):
	pass
```

-	用途：按照`key`指定类型返回表示泛型类的专门化对象
	-	实现*PEP 484*规定的泛型类语法
	-	查找基于对象自身
	-	主要被保留用于静态类型提示，不鼓励其他尝试使用
	-	方法默认、隐式为类方法，不需要`classmethod`封装

-	参数
	-	`cls`：当前类
	-	`key`：类型

##	模拟可调用对象

###	`__call__`

```python
def object.__call__(self[,args...]):
	pass
```

-	用途：实例作为函数被调用时被调用
	-	若定义此方法`x(arg1, arg2, ...)`等价于
		`x.__call__(arg1, args2, ...)`

##	模拟容器类型

-	`collections.abc.MutableMapping`为抽象基类
	-	其实现基本方法集`__getitem__`、`__setitem__`、
		`__delitem__`、`keys()`
	-	可以方法继承、扩展、实现自定义映射类

###	`__len__`

```python
def object.__len__(self):
	pass
```

-	用途：计算、返回实例长度
	-	若对象未定义`__bool__`，以`__len__`是否返回非0作为
		布尔运算结果

-	返回值：非负整形
	-	钩子函数：`len()`

> - CPython：要求长度最大为`sys.maxsize`，否则某些特征可能
	会`raise OverflowError`

###	`__length_hint__`

```python
def object.__length_hist__(self):
	pass
```

-	用途：返回对象长度**估计值**
	-	存粹为优化性能，不要求正确无误

-	返回值：非负整形
	-	钩子函数：`operator.length_hint()`

###	`__getitem__`

```python
def object.__getitem__(self, key):
	pass
```

-	用途：实现根据索引取值

-	参数
	-	序列`key`：整数、切片对象
		-	`key`类型不正确将`raise TypeError`
		-	`key`在实例有效范围外将`raise IndexError`

	-	映射`key`：可hash对象
		-	`key`不存在将`raise KeyError`

-	返回值：`self[key]`

###	`__setitem__`

```python
def object.__setitem__(self, key, value):
	pass
```

-	用途：实现根据索引赋值

-	参数：同`__geitem__`

###	`__delitem__`

```python
def object.__delitem(self, key):
	pass
```

-	用途：实现删除索引对应项

-	参数：同`__getitem__`

###	`__missing__`

```python
def object.__missing__(self, key):
	pass
```

-	用途：`__getitem__`无法找到映射中键时调用

###	`__reversed__`

```python
def object.__iter__(self):
	pass
```

-	用途：为容器类创建逆向迭代器

-	返回值：逆向迭代对象
	-	内置钩子函数：`reversed()`

####	说明

-	若未提供`__reversed__`方法，`reversed`函数将回退到使用
	**序列协议**：`__len__`、`__getitem__`

-	支持序列协议的对象应该仅在能够提供比`reversed`更高效实现
	时才提供`__reversed__`方法

###	`__contains__`

```python
def object.__contains__(self, item):
	pass
```

-	用途：实现成员检测
	-	若`item`是`self`成员则返回`True`、否则返回`False`
	-	对映射应检查键

-	返回值：布尔值
	-	钩子运算：`in`

####	说明

-	若未提供`__contains__`方法，成员检测将依次尝试
	-	通过`__iter__`进行迭代
	-	使用`__getitem__`旧式序列迭代协议

-	容器对象可以提供更有效率的实现

##	模拟数字

###	数字运算

定义以下方法即可模拟数字类型

-	特定类型数值类型不支持的运算应保持未定义状态
-	若不支持与提供的参数进行运算，应返回`NotImplemented`

```python
def object.__add__(self, other):
	# `+`
def object.__sub__(self, other):
	# `-`
def object.__mul__(self, other):
	# `*`
def object.__matmul__(self, other):
	# `@`
def object.__truediv__(self, other):
	# `/`
def object.__floordiv__(self, other):
	# `//`
def object.__mod__(self, other):
	# `%`
def object.__divmod__(self, other):
	# `divmod()`
def object.__pow__(self, other[, modulo=1]):
	# `pow()`/`**`
	# 若要支持三元版本内置`pow()`函数，应该接受可选的第三个
		# 参数
def object.__lshift__(self, other):
	# `<<`
def object.__rshift__(self, other):
	# `>>`
def object.__and__(self, other):
	# `&`
def object.__or__(self, other):
	# `|`
def object.__xor__(self, other):
	# `~`
```

###	反射二进制算术运算

以下成员函数仅在**左操作数不支持相应运算**、
**且两操作数类型不同时**被调用

-	实例作为作为相应运算的右操作数

-	若右操作数类型为左操作数类型子类，且字类提供如下反射方法
	-	右操作数反射方法优先于左操作数非反射方法被调用
	-	允许子类覆盖祖先类运算符

-	三元版`pow()`不会尝试调用`__rpow__`（转换规则太复杂）

```python
def object.__radd__(self, other):
	# `+`
def object.__rsub__(self, other):
	# `-`
def object.__rmul__(self, other):
	# `*`
def object.__rmatmul__(self, other):
	# `@`
def object.__rtruediv__(self, other):
	# `/`
def object.__rfloordiv__(self, other):
	# `//`
def object.__rmod__(self, other):
	# `%`
def object.__rdivmod__(self, other):
	# `divmod()`
def object.__rpow__(self, other[, modulo=1]):
	# `pow()`/`**`
	# 若要支持三元版本内置`pow()`函数，应该接受可选的第三个
		# 参数
def object.__rlshift__(self, other):
	# `<<`
def object.__rrshift__(self, other):
	# `>>`
def object.__rand__(self, other):
	# `&`
def object.__ror__(self, other):
	# `|`
def object.__rxor__(self, other):
	# `~`
```

###	扩展算术赋值

实现以下方法实现扩展算数赋值

-	以下方法应该尝试对自身进行操作
	-	修改`self`、返回结果（不一定为`self`）

-	若方法未定义，相应扩展算数赋值将回退到普通方法中

-	某些情况下，扩展赋值可导致未预期错误

```python
def object.__iadd__(self, other):
	# `+=`
def object.__isub__(self, other):
	# `-=`
def object.__imul__(self, other):
	# `*=`
def object.__imatmul__(self, other):
	# `@=`
def object.__itruediv__(self, other):
	# `/=`
def object.__ifloordiv__(self, other):
	# `//=`
def object.__imod__(self, other):
	# `%=`
def object.__ipow__(self, other[, modulo=1]):
	# `**=`
def object.__ilshift__(self, other):
	# `<<=`
def object.__irshift__(self, other):
	# `>>=`
def object.__iand__(self, other):
	# `&=`
def object.__ior__(self, other):
	# `|=`
def object.__ixor__(self, other):
	# `~=`
```

###	一元算术运算

```python
def object.__neg__(self):
	# `-`
def object.__pos__(self):
	# `+`
def object.__abs__(self):
	# `abs()`
def object.__invert__(self):
	# `~`
```

###	类型转换运算

```python
def object.__complex__(self):
	# `complex()`
def object.__int__(self):
	# `int()`
def object.__float__(self):
	# `float()`
```

###	整数

```python
def object.__index__(self):
	pass
```

-	存在此方法表明对象属于整数类型
	-	必须返回整数
	-	为保证以一致性，同时也应该定义`__int__()`，两者返回
		相同值

-	调用此方法以实现`operator.index()`、或需要无损的转换为
	整数对象
	-	作为索引、切片参数
	-	作为`bin()`、`hex()`、`oct()`函数参数

###	精度运算

```python
def object.__round__(self[, ndigits]):
	# `round()`
def object.__trunc__(self):
	# `math.trunc()`
def object.__floor__(self):
	# `math.floor()`
def object.__ceil__(self):
	# `math.ceil()`
```

-	返回值：除`__round__`中给出`ndigits`参数外，都应该为
	原对象截断为`Integral`（通常为`int`）

-	若未定义`__int__`，则`int`回退到`__trunc__`

##	元属性查找

-	元属性查找通常会绕过`__getattribute__`方法，甚至包括元类

	```python
	class Meta(type):
		def __getattribute__(*args):
			print("Metaclass getattribute invoked")
			return type.__getattribute__(*args)

	class C(object, metaclass=Meta):
		def __len__(self):
			return 10
		def __getattribute__(*args):
			print("Class getattribute invoked")
			return object.__geattribute__(*args)

	if __name__ == "__main__":
		c = C()
		c.__len__()
			# 通过实例显式调用
			# 输出`Class getattribute invoked\n10"
		type(c).__len__(c)
			# 通过类型显式调用
			# 输出`Metaclass getattribute invoked\n10"
		len(c)
			# 隐式查找
			# 输出`10`
	```

	-	为解释器内部速度优化提供了显著空间
	-	但是牺牲了处理特殊元属性时的灵活性
		-	特殊元属性必须设置在类对象本身上以便始终一致地
			由解释器发起调用

-	隐式调用元属性仅保证**元属性定义在对象类型中**能正确发挥
	作用

	```python
	class C:
		pass

	if __name__ == "__main__":
		c = C()
		c.__len__() = lambda: 5
		len(c)
			# `rasie TypeError`
	```
	-	元属性定义在实例字典中会引发异常
	-	若元属性的隐式查找过程使用了传统查找过程，会在对类型
		对象本身发起调用时失败
	-	可以通过在查找元属性时绕过实例避免

		```python
		>>> type(1).__hash__(1) == hash(1)
		>>> type(int).__hash__(int) == hash(int)
		```

##	上下文管理器协议

上下文管理器：定义了在执行`with`语句时要建立的运行时上下文
的对象

-	上下文管理器为执行代码块，处理进入、退出运行时所需上下文
	-	通常使用`with`语句调用
	-	也可以直接调用协议中方法方法

-	典型用法
	-	保存、恢复各种全局状态
	-	锁、解锁资源：避免死锁
	-	关闭打开的文件：自动控制资源释放

> - 可利用`contextlib`模块方便实现上下文管理器协议

###	`__enter__`

```python
def contextmanager.__enter__(self):
	pass
```

-	用途：创建、进入与当前对象相关的运行时上下文
	-	在执行`with`语句块前设置运行时上下文

-	返回值
	-	`with`子句绑定方法返回值到`as`子句中指定的目标，如果
		方法返回值

###	`__exit__`

```python
def contextmanger.__exit__(self, exc_type, exc_value, traceback):
	pass
```

-	用途：销毁、退出关联到此对象的运行时上下文
	-	`with`语句块结束后，`__exit__`方法触发进行清理工作
	-	不论`with`代码块中发生什么，即使是出现异常，
		`__exit__`控制流也会执行完

-	参数：描述了导致上下文退出的异常，正常退出则各参数为
	`None`
	-	`exc_type`
	-	`exc_value`
	-	`traceback`

-	返回值：布尔值
	-	若上下文因异常退出
		-	希望方法屏蔽此异常（避免传播），应该返回真值，
			异常被清空
		-	否则异常在退出此方法时将按照正常流程处理
	-	方法中不应该重新引发被传入的异常，这是调用者的责任

###	例

```python
from socket import socket, AF_INET, SOCK_STREAM

class LazyConnection:
	def __init__(self, address, family=AF_INET, type=SOCK_STREAM):
		self.address = address
		self.family = family
		self.type = type
		self.connections = []

	def __enter__(self):
		sock = socket(self.family, self.type)
		sock.connect(self.address)
		self.connections.append(sock)
		return self.sock

	def __exit__(self, exc_ty, exc_val, tb):
		self.connections.pop().close()

from functools import partial

def test():
	conn = LazyConnection("www.python.org", 80))
	with conn as s1:
		# `conn.__enter___()` executes: connection opened
		s.send(b"GET /index.html HTTP/1.0\r\n")
		s.send(b"Host: www.python.org\r\n")
		s.send(b"\r\n")
		resp = b"".join(iter(partial(s.recv, 8192), b""))
		# `conn.__exit__()` executes: connection closed

		with conn as s2:
			# 此版本`LasyConnection`可以看作是连接工厂
			# 使用列表构造栈管理连接，允许嵌套使用
			pass
```

##	迭代器协议

-	可迭代对象：实现`__iter__`方法的对象
-	迭代器对象：同时实现`__next__`方法的可迭代对象

> - 使用`collections.abc`模块判断对象类型

###	`__iter__`

```python
def object.__iter__(self):
	pass
```

-	用途：创建迭代器对象，**不负责产生、返回迭代器元素**
	-	容器对象要提供迭代须实现此方法
		-	容器支持不同迭代类型，可以提供额外方法专门请求
			不同迭代类型迭代器
	-	迭代对象本身需要实现此方法，返回对象自身
		-	允许容器、迭代器均可配合`for...in...`语句使用

-	返回值：迭代器对象
	-	映射类型应该逐个迭代容器中键
	-	内置钩子函数：`iter()`

> - 此方法对应Python/C API中python对象类型结构体中
	`tp_iter`槽位

###	`__next__`

```python
def object.__next__():
	pass
```

-	用途：从迭代器中返回下一项
	-	若没有项可以返回，则`raise StopIteration`
	-	一旦引发`raise StopIteration`，对后续调用必须一直
		引发同样的异常，否则此行为特性无法正常使用

-	返回值：迭代器对象中下个元素
	-	映射类型返回容器中键
	-	内置钩子函数：`next()`

> - 此方法对应Python/C API中python对象类型结构体中
	`tp_iternext`槽位

##	协程/异步

###	`__await__`

```python
def object.__await__(self):
	pass
```

-	用途：用于实现**可等待对象**

-	返回值：迭代器
	-	钩子运算：`await`

> - `asyncio.Future`实现此方法以与`await`表达式兼容

####	*Awaitable Objects*

可等待对象：异步调用句柄，**等待结果应为迭代器**

-	主要是实现`__await__`方法对象
	-	从`async def`函数返回的协程对象

-	`type.coroutine()`、`asyncio.coroutine()`装饰的生成器
	返回的生成器迭代器对象也属于可等待对象，但其未实现
	`__await__`

> - 协程对象参见*cs_python/py3ref/dm_gfuncs*
> - py3.7前多次`await`可等待对象返回`None`，之后报错

###	异步迭代器协议

-	异步迭代器常用于`async for`语句中

> - 其他参见迭代器协议

####	`__aiter__`

```python
def object.__aiter__(self):
	pass
```

-	用途：返回异步迭代器对象，**不负责产生、返回迭代器元素**
	-	返回其他任何对象都将`raise TypeError`

> - 其他参见`__iter__`方法

####	`__anext__`

```python
async def object.__anext__(self):
	pass
```

-	返回：从异步迭代器返回下个结果值
	-	迭代结束时应该`raise StopAsyncIteration`

-	用途
	-	在其中调用异步代码

> - 其他参见`__next__`方法

####	例

```python
class Reader:
	async def readline(self):
		pass
	def __aiter__(self):
		return self
	async def __anext__(self):
		val = await self.readline()
		if val == "b":
			raise StopAsyncIteration
		return val
```

###	异步上下文管理器协议

-	异步上下文管理器常用于`async with`**异步**语句中

> - 其他参见上下文管理器协议

####	`__aenter__`

```python
async def object.__aenter__(self):
	pass
```

-	用途：异步创建、进入关联当前对象的上下文执行环境
	-	由`async def`定义为协程函数，即在创建上下文执行环境
		时可以被挂起

-	返回：可等待对象

> - 其他参见`__enter__`

####	`__aexit__`

```python
async def object.__aexit__(self):
	pass
```

-	用途：异步销毁、退出关联当前对象的上下文执行环境
	-	由`async def`定义为协程函数，即在销毁上下文执行环境
		时可以被挂起

-	返回：可等待对象

> - 其他参见`__exit__`函数

