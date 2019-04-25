#	类元信息

类元信息：python类中具有特殊名称的方法，实现**由特殊语法**
所引发的特定操作

-	python实现**操作符重载**的方式
	-	允许每个类自行定义基于操作符的特定行为
	-	特定操作包括python内置的**钩子函数**

-	没有定义适当方法的情况下，尝试执行操作将引发异常
	`raise AttributeError`、`raise TypeError`

-	将特殊方法设为`None`表示对应操作不可用

-	对大部分元信息，python有相应的钩子函数调用该方法，用于
	方便访问元信息

-	但是这些钩子函数也不能简单的看作是**直接访问**元信息，
	其可能比直接访问要**更底层**

	-	所以推荐使用钩子函数，而不要直接访问对象元信息
	-	案例可以查看<#属性的代理访问>

-	类继承会获得基类的所有方法
	-	类里面的方法其实真的不是给类使用的，而是给实例使用
	-	类自身使用的方法是元类中的方法

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
def object.__new__(cls[, *args, **kwargs]):
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

-	这里是给类实例绑定方法的地方????

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

```c
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
	-	相等的对象**理应**具有相同hash值
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
	-	若指定方法没有相应实现，富比较方法可能返回单例对象
		`NotImplemented`

###	说明

-	默认情况下，`__ne__`会委托给`__eq__`，并将结果取反，除非
	结果为`NotImplemented`

-	比较运算符之间没有其他隐含关系
	-	`x < y or x == y`为真不意味着`x <= y`
	-	要根据单根运算自动生成排序操作可以利用
		`functools.total_ordering()`简化

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

###	`__len__`

-	用途：计算、返回实例长度

-	返回值：整形
	-	钩子函数：`len`

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

-	用途：默认属性访问引发`AttributeError`而失败时调用
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

-	调用`__getattr__`应该是`.`运算符中逻辑

-	`__getattr__`甚至不是`object`具有的
	`<wrapper_descriptor>`

-	相较于`__getattribute__`其实更常用，因为修改**所有**对
	对对象的访问逻辑没啥价值

###	`__getattribute__`

```python
def __getattribute(self, key):
	"Emulate type_getattro() in Objects/typeobject.c"
	v = object.__getattribute__(self, key)
	if hasattr(v, "__get__"):
		return v.__get__(None, self)
	return v
```

-	用途：对**实例属性**访问时无条件被调用
	-	已定义`__getattr__`不会被调用，除非`__getattribute__`
		显式调用、或`raise AtttributeError`
	-	为**避免方法中无限递归**，实现总应该调用具有相同名称
		基类方法访问所需要的属性
	-	作为**通过特定语法、内置函数隐式调用的结果**情况下，
		查找特殊方法时仍可能被跳过
	#todo

-	返回值：找到的属性值、或`raise AttributeError`

####	说明

-	`object.__getattribute__`是`wrapper_descriptor`C实现的函数

-	`__getattribute__`只对新式类实例可用

-	其在实例、类中类型不同
	-	类中`__getattribute__`是`wrapper_descriptor`类型，
		相当于函数，没有绑定实例
	-	实例中`__getattribute__`是`method-wrapper`类型，相当
		于方法，绑定当前实例

-	其对实例、类行为不同
	-	访问实例的任何属性，即使用`.`时，`__getattribute__`
		作为绑定方法都会被调用
	-	访问类属性时，调用的应该是元类的`__getattribute__`

-	其是描述器调用的关键
	-	重写`__getattribute__`方法可以阻止描述器的调用

####	钩子函数

-	`.`运算符：首先调用`__getattribute__`，若无访问结果，
	调用`__getattr__`

-	`getattr`：类似`.`运算符，只是可以捕获异常，设置默认
	返回值

-	`hasattr`：内部调用`getattr`，根据`raise Exception`判断
	属性是否存在

	-	可以通过`@property.getter`中`raise AttributeError`
		使得属性看起来不存在

	-	内部有更多`boilerplate`相较于`getattr`更慢

	-	所以，按照字面意思使用不需要考虑过多

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
		-	若正常查账`__getattribute__`无法在模块中找到某个
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

	-	如果只是想简单的自定义某个类的单个属性访问的话，使用
		`@porperty`更方便

###	描述器方法

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

```python
def __get__(self, instance, cls):
	if instance is None:
		# 装饰器类只能作为类属性，需要考虑通过类直接访问
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

```python
def __set__(self, instance, value):
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

-	实例绑定中，描述器调用的**优先级**取决于描述器定义的方法
	-	优先级：资料描述器 > 实例字典属性 > 非资料描述器
	-	实例属性会重载非资料描述器
	-	实例属性和资料描述器同名时，优先访问描述器，否则优先
		访问属性

-	只读资料描述器：`__set__`中`raise AttributeError`得到

####	Python设计

-	python方法都实现为非资料描述器，则实例可以重定义、重载
	方法
	-	`staticmethod`：静态方法
	-	`classmethod`：类方法
	-	实例方法

-	`@property`方法被实现为资料描述器

> - `super`、属性、实例的实现的依赖于描述器i

###	特殊描述器类

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

####	`@property`

> - 代码是C实现，这里是python模拟，和`help`结果不同

```python
class Property(object):
	"Emulate PyProperty_Type() in Objects/descrobject.c"

	def __init__(self, fget=None, fset=None, fdel=None, doc=None):
		self.fget = fget
		self.fset = fset
		self.fdel = fdel
		if doc is None and fget is not None:
			doc = fget.__doc__
		self.__doc__ = doc

	def __get__(self, obj, objtype=None):
		if obj is None:
			return self
		if self.fget is None:
			raise AttributeError("unreadable attribute")
		return self.fget(obj)

	def __set__(self, obj, value):
		if self.fset is None:
			raise AttributeError("can't set attribute")
		self.fset(obj, value)

	def __delete__(self, obj):
		if self.fdel is None:
			raise AttributeError("can't delete attribute")
		self.fdel(obj)

	def getter(self, fget):
		return type(self)(fget, self.fset, self.fdel, self.__doc__)
		# 返回描述器，可省略

	def setter(self, fset):
		return type(self)(self.fget, fset, self.fdel, self.__doc__)
		# 返回更新`fset`的描述器，同名所以覆盖前者

	def deleter(self, fdel):
		return type(self)(self.fget, self.fset, fdel, self.__doc__)
```

-	`@property`同样是描述器类，接受方法返回同名资料描述器

####	`super`

```python
class super:
	super()
		# 等同于：`super(__class, <first_argument>)`
		# `<first_argument>`常常就是`self`
	super(type)
		# 返回：未绑定super对象，需要`__get__`绑定
	super(type, obj)
		# 返回：已绑定super对象，要求`isinstance(obj,type)`
	super(type, type2)
		# 返回：已绑定super对象，要求`issubclass(type2, type)`
		# 此时调用方法返回是函数，不是绑定方法，不会默认传入
			# `type2`作为首个参数

	def __get__(self, obj, type=None):
		

def super(cls, inst/subcls):
    mro = inst.__class__.mro()
	mro = subcls.mro()
    return mro[mro.index(cls) + 1]
```

-	参数
	-	第一个参数：MRO列表中定位，确定起始调用类
	-	第二个参数：**提供MRO列表**
		-	类：直接传递MRO列表
		-	实例：传递所属类的MRO列表

-	返回：MRO列表中第一个参数的下个类
	-	所以是可以通过在某个类中方法中使用超类调用`super`
		跳过某些类中方法
	-	只有MRO列表中每个类中的方法都`super()`调用，才能保证
		列表中所有类的该方法都被调用

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
def object.__init_subclass__(cls):
	pass
```

-	用途：派生类继承父类时，基类的`__init_subclas__`被调用
	-	可以用于编写能够改变子类行为的类
	-	类似类装饰器，但是类装饰其影响其应用的类，而
		`__init_subclass__`影响基类所有派生子类
	-	默认实现：无行为、只有一个参数`cls`

-	参数：默认实现无参数，可以覆盖为自定义参数

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

####	说明

-	方法默认为类方法，不需要`classmethod`封装
	-	参数`cls`指向新的子类

###	元类

##	字典相关

###	`__getitem__`

###	`__setitem__`

###	`__delitem__`

###	`__hash__`

##	`with`语句（上下文管理协议）

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
			# 此版本`LasyConnection`可以看作是连接工厂J
			# 使用列表构造栈管理连接，允许嵌套使用
			pass
```

###	`__enter__`

-	`with`语句出现时，对象的`__enter__`方法被触发，返回值被
	赋给`as`后声明的变量，然后`with`语句块里的代码开始执行

###	`__exit__`

-	`with`语句块结束后，`__exit__`方法触发进行清理工作
	-	不论`with`代码块中发生什么，即使是出现异常，
		`__exit__`控制流也会执行完

-	方法的第三个参数`exc_val`包含异常类型、异常值、和
	回溯信息，可以自定义方法决定如何利用异常信息
	-	返回`True`， 异常会被清空

-	使用`__enter__`、`__exit__`、`with`自动控制资源释放，
	有效避免死锁


