#	Python类、对象

##	类属性控制

###	封装属性

python没有访问控制，不依赖语言特性封装数据，而是遵循一定属性
、方法命名规约达到效果

-	任何以单`_`开头的名字都应该是内部实现
	-	此约定同样适用于模块名、模块级别函数
	-	python不会真的防止访问内部名称，但是这样会导致脆弱的
		代码

-	以`__`开头的名称会使名称变为其他形式
	-	目的是为了防止继承，这样的属性无法通过继承覆盖
	-	如果清楚代码会涉及子类，且应该在子类中隐藏起来，考虑
		使用双下划线开头

-	以单`_`结尾，避免定义的变量和保留关键字冲突

```python
class A:
	def __init__(self):
		self._internal = 0
			# private attr
		self.public = 1

	def public_method(self):
		pass

	def _internal_method(self):
		# private method
		pass

class B:
	def __init__(self):
		self.__private = 0
			# 将会被重命名为`_B__private`

	def __private_method(self):
		# 将会被重命名为`_B__private_method`
		pass

	def public_method(self):
		self.__private_method()

def C(B):
	def __init__(self):
		super().__init__()
		self.__private = 1
			# 不会覆盖`B.__private`
			# 会被重命名为`C._C__private`

	def __private_method(self):
		# 不会覆盖`B.__private_method`
		# 会被重命名为`C._C__private_method`
		pass
```

###	`@property`可管理属性

使用`@property`装饰器可以为类的属性增加其他处理逻辑，如：
类型检查、合法性验证

-	`@property`装饰器类将getter-like方法转变为属性，然后设置
	3个方法名开头装饰器分别装饰其他两个**同名方法**

	-	`@method.setter`：将setter-like方法转换成`method`
		属性，在对`.method`赋值时将被调用
		-	不设置时会将方法转换为“只读属性”
	-	`@method.deleter`：将deleter-like方法转换为`method`
		属性，在`del(.method)`时将被调用
	-	`@method.getter`：将getter-like方法转换为`method`
		属性，在访问`.method`时调用，这个一般只在子类仅修改
		父类property属性的getter方法时使用

-	`@property`装饰的属性和普通属性没有区别

	-	但是在访问时会自动触发`getter`、`setter`、`deleter`
	-	property属性就是绑定有这些方法的类实例，通常不需要
		手动调用这些方法
		-	`instance.method.fget`
		-	`instance.method.fset`
		-	`instance.method.fdel`
	-	没有必要却使用`@property`时会迷惑读者、使代码臃肿、
		降低代码效率
	-	有时候使用get/set方法可能有更好的兼容性

-	应用`@property`时，仍然需要在类中存储底层数据，在
	`getter`、`setter`会对其进行处理

	-	初始化时，不应该直接设置底层数据属性，会绕过`setter`
		的参数检查
	-	当然，有的写property属性没有对应的底层数据属性，是在
		需要的时候计算出来的，当然也没有对应的`setter`方法
		实现

-	可以直接使用已有getter-like、setter-like方法创建
	property类实例，定义property属性

	-	property属性就是一个类，为其实现`setter`、`getter`、
		`deleter`方法，然后作为**类属性**即可
	-	因为本身直接使用`@property`装饰方法也是相当于直接
		设置的是**类属性**
	-	其内部有可能是使用的描述器将值记录在实例底层字典
		`__dict__`中

```python
class Student(object):

	def __init__(self, value):
		self.birth = value
			# 使用`self.birth`而不是`self._birth`，保证即使
				# 实在初始化时仍然进行参数检查

	@property
		# 将一个getter-like方法变为属性
		# `@property`同时会创建装饰器`@method.setter`
	def birth(self):
		return self._birth

	@birth.setter
		# `@property`对应，将setter-like方法变为属性
	def birth(self, value):
		if not instance(value, int):
			raise ValueError("birth must be an integer")
		if value < 1900 or value > 2020:
			raise ValueError("birth must between 1900 ~ 2020")
		self._birth = value

	@birth.deleter
		# 同`@property`对应，在`del`时调用
	def birth(self):
		del(self._age)
		del(self._birth)

	@property
		# 只设置`@property`而没有设置对应`@birth.setter`
		# 这样`birth`就成了只读属性
	def age(self):
		return 2018 - self._birth

	def get_first_name(self):
		return self._first_name
	
	def set_first_name(self):
		if not instance(value, str):
			raise TypeError("expected a string")
		self._first_name = value

	def del_first_name(self):
		raise AttributeError("can't delete attribute")

	name = property(get_first_name,
		set_first_name,
		del_first_name)
		# 在已有getter-like、setter-like方法上创建property
		# 注意：这里就是应该定义类属性，本身使用`@property`
			# 装饰器也是相当于创建类属性
```

###	`classmethod`类方法

###	`staticmethod`静态方法

###	构造实例

-	`@classmethod`类方法可以实现多个构造器

-	`__new__`可以方法绕过`__init__`，创建未初始化的实例

	-	这种方法可以用于**反序列**对象，从字符串反序列构造
		符合要求的对象

```python
import time

class Date:
	def __init__(self, year, month, day):
		self.year = year
		self.month = month
		self.day = day

	@classmethod
	def today(cls):
		# 类方法接受`class`作为第一个参数
		# 和普通属性方法一样能够继承
		t = time.localtime()
		return cls(t.tm_year, t.tm_mon, t.tm_today)

	@classmethod
	def today2(cls):
		d = cls.__new__(cls)
			# 使用`__new__`绕过`__init__`创建新实例，不对
				# 实例进行初始化
		t = localtime()
		d.year = t.tm_year
		d.month = t.tm_mon
		d.day = t.tm_mday
		return d

class NewData(Date):
	pass

def test():
	c = Date.today()
	d = NewDate.today()
```

##	类元信息

python对象内`__`开头的方法、属性称为类的元信息

-	对大部分元信息，python有相应的钩子函数，可以**看作**是
	直接访问对象内部相应方法、属性

-	但是这些钩子函数也不能简单的看作是**直接访问**元信息，
	其可能比直接访问要**更底层**

	-	所以推荐使用钩子函数，而不要直接访问对象元信息
	-	案例可以查看<#属性的代理访问>

###	实例创建

####	`__new__`

-	钩子函数：`cls()`

	-	不全是，至少`__init__`方法应该不是在`__new__`中
		调用的，所以`cls()`不只是调用了`__new__`方法
	-	所以若要真正跳过`__init__`方法，是应该直接调用
		`__new__`方法，而不是修改`__new__`然后`cls()`实例化
	-	但是`__new__`确实**返回类实例**

-	`__new__`创建的未初始化实例若跳过了`__init__`方法

	-	`cls.__new__`中实例从类中**获取**实例方法，此不需要
		再次设置
	-	`__init__`方法中设置的实例属性则需要手动设置
	-	设置实例属性时，最好不要直接访问、操作实例底层字典
		`__dict__`，避免破坏使用`__slots__`、`@property`、
		描述器类等高级技术的代码，除非很清楚操作结果

-	`__new__`方法还有一个特殊的地方，`__new__`不需要使用
	`@classmethod`装饰，但是这个其第一个参数是`cls`，是一个
	类方法

####	`__init__`

-	**初始化**类实例

	-	这个显然不应该是`cls()`直接调用的函数，因为这个根本
		没有返回值
	-	但这个函数其实泛用性更强，创建实例后被调用、设置实例
		属性，很大程度可以看作是**创建**新实例的函数

####	`__del__`

-	钩子函数：`del`

####	`__prepare__`

在所有类定义开始执行前被调用，用于创建类命名空间，一般这个
方法只是简单的返回一个字典或其他映射对象

###	输出相关

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

####	`__str__`

-	钩子函数：`str`

	-	`print`函数对于非`str`类型会隐式调用`str`/`__str__`

-	将实例转换为一个字符串

####	`__repr__`

-	钩子函数：`repr`

	-	**交互环境**下直接“执行”变量的结果

-	实例的代码表示形式

	-	其生成的文本字符串标准的做法是需要让
		`eval(repr(x))==x`为真，至少也需要创建有用的文本
		表示，并使用`<>`扩起
	-	通常用于重新构造实例

-	格式化代码中`!r`指明输出使用`__repr__`而不是默认
	`__str___`

	-	如果`__str__`没有定义，会使用`__repr__`代替输出

####	`__format__`

-	钩子函数：`format`

-	自定义类的字符串格式

###	内部信息

####	`__dict__`

-	钩子函数：`vars`、`dir`（部分）

	-	`vars`是真正对应的钩子函数，返回键值对
	-	`dir`执行过程中会访问`__dict__`、`__class__`，而且
		只返回keys

-	**对象**底层字典，存储对象属性、方法

	-	注意区分：类属性、实例属性、父类属性，`__dict__`只
		包括**当前实例**属性、方法
	-	返回结果是`dir`结果的子集

-	在大部分情况下会自动更新，比如`setattr`函数时，或者
	说实例的属性、方法更新就是`__dict__`的变动

	-	但是一般情况下不要直接访问`__dict__`，除非真的清楚
		所有细节，如果类使用了`cls.__slots__`、`@property`、
		描述器类等高级技术时代码可能会被破坏
	-	尽量使用`setattr`函数，让python控制其更新

####	`__class__`

-	钩子函数：`type`

-	对象类型

####	`__len__`

-	钩子函数：`len`

-	对象长度

###	`with`语句（上下文管理协议）

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

####	`__enter__`

-	`with`语句出现时，对象的`__enter__`方法被触发，返回值被
	赋给`as`后声明的变量，然后`with`语句块里的代码开始执行

####	`__exit__`

-	`with`语句块结束后，`__exit__`方法触发进行清理工作
	-	不论`with`代码块中发生什么，即使是出现异常，
		`__exit__`控制流也会执行完

-	方法的第三个参数`exc_val`包含异常类型、异常值、和
	回溯信息，可以自定义方法决定如何利用异常信息
	-	返回`True`， 异常会被清空

-	使用`__enter__`、`__exit__`、`with`自动控制资源释放，
	有效避免死锁

###	描述器类

描述器就是实现了三个核心的属性访问操作get、set、delete的类，
分别对应`__get__`、`__set__`、`__delete__`特殊的方法

-	描述器只能定义为类属性，不能定义为实例属性

	-	描述器类虽然是类属性，但是其在实现`__set__`方法后，
		其优先级高于类实例属性
	-	仅仅实现`__get__`方法，描述器类属性的优先级低于
		类实例属性
	-	对类同名属性的访问会优先触发描述器类属性

-	其接受一个实例作为输入，之后相应的操作实例底层字典

	-	这样存储值，虽然描述器类是作为类属性，看起来仍然对
		每个实例取不同值

-	所有对描述器属性的访问会被`__get__`、`__set__`、
	`__delete__`方法捕获到

-	如果只是像简单的自定义某个类的单个属性访问的话，使用
	`@porperty`更容易

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

####	`__get__`

-	**访问**描述器类时被调用，接受三个参数
-	当描述器类中只实现`__get__`方法时，描述器对类实例会有
	更弱绑定，只有类示例`__dict__`没有对应属性时才能在访问
	属性时被触发
-	当然，如果仅仅只是从底层实例字典中获取属性值，`__get__`
	方法不用实现

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

对描述器类**赋值**时被调用，接受三个参数

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

“删除”描述器类时调用，接受两个参数

```python
def __delete__(self, instance):
	if instance is None:
		pass
	else:
		del instance.__dict__[self.name]
		# 操作实例底层`__dict__`
```

###	属性相关

####	`__getattribute__`

访问对象的**任何**方法、属性（即只要使用了`.`运算符，无论
方法、属性是否存在）都会被触发

-	钩子函数：`getattr`

-	**最基类**的`__getattribute__`方法中，如果方法、属性在
	当前实例中不存在，会再调用`__getattr__`函数

####	`__getattr__`

-	`__getattr__`方法在访问对象属性、方法**不存在**时被调用
	，这个应该是写在`__getattribute__`里的逻辑

-	相较于`__getattribute__`其实更常用，因为修改**所有**对
	对对象的访问逻辑没啥价值

####	`__setattr__`

-	钩子函数：`setattr`

####	`__delattr__`

-	钩子函数：`delattr`

###	字典相关

####	`__getitem__`

####	`__setitem__`

####	`__delitem__`

###	比较相关

####	`__eq__`

-	钩子：`==`/`!=`

-	允许类支持`==`/`!=`运算

####	`__gt__`/`__ge__`

-	钩子：`>`/`>=`

-	允许类支持`>`/`>=`运算

####	`__lt__`/`__le__`

-	钩子：`<`/`<=`

-	允许类支持`<`/`<=`运算

####	`@functools.total_ordering`

```python
from functools import total_ordering

class Room:
	def __init__(self, name, length, width):
		self.name = name
		self.length = length
		self.width = width
		self.square_feet = self.length * self.width

@total_ordering
	# `total_ordering`允许只定义`__eq__`和其他中的一个，其他
		# 的方法由装饰器自动填充
class House:
	def __init__(self, name, style):
		self.name = name
		self.style = style
		self.rooms = list()

	@property
	def living_space_footage(self):
		return sum(r.square_feet for r in self.rooms)

	def add_room(self, room):
		self.rooms.append(room)

	def __str__(str):
		return "{}: {} squre foot {}".format(
			self.name,
			self.living_space_footage,
			self.style)

	def __eq__(self, other):
		return self.living_space_footage == other.living_space_footage

	def __lt__(self, other):
		return self.living_space_footage < other.living_space_footage
```

###	类属性

####	`cls.__slots___`

-	定义类属性`__slots__`后，python会为实例属性使用紧凑内部
	表示
	-	实例属性使用固定大小、很小的数组构建，而不是为每个
		实例定义字典
	-	在`__slots__`列出的属性名在内部映射到数组指定小标上
	-	类似于R中`factor`类型、C中`enum`类型

-	因此不能再给实例添加新的属性，只能使用在`__slots__`中
	定义的属性名

-	但是python很多特性依赖于普通的依赖字典的实现，定义
	`__slots__`的类不再支持普通类的特性，如：多继承
	-	大多数情况下，应该旨在经常使用到作为数据结构的类上
		定义`__slots__`
	-	不应该把`__slots__`作为防止用户给实例增加新属性的
		封装工具

```python
class Date:
	__slots__ = ["year", "month", "day"]
	def __init__(self, year, month, day):
		self.year = year
		self.month = month
		self.day = day
```

##	继承

###	`super`

`super`函数可以用于调用父类（超类）方法

-	在`__init__`方法中保证父类正确初始化
-	覆盖python特殊方法
-	尽量使用`super`调用父类方法，而不是直接使用父类名称，
	避免多继承中出现问题，如：超类方法多次调用

>	可参见`puzzles`

```python
class A:
	def __init__(self):
		self.x = 0

	def spam(self):
		print("A.spam")

class B(A):
	def __init__(self):
		super().__init__()
			# 在`__init__`中保证父类正确初始化
		self._obj = obj

	def spam(self):
		print("B.spam")
		super().spam()

	def __getattr__(self, name):
		return getattr(self._obj, name)

	def __setattr__(self, name, value):
		if name.startswith("_"):
			super().__setattr__(name, value)
				# `super`用于覆盖python特殊方法
		else:
			setattr(self._obj, name, value)
```

###	`@property`

在子类中扩展property属性和一般的方法、属性有差别，需要遵守
一定的规则，否则property属性无法正常工作

-	`@property`直接装饰，必须覆盖property属性的所有方法，
	否则没有重新定义的property方法不能正常工作

	-	因为property属性其实是getter、setter、deleter方法的
		集合
	-	直接使用`@property`装饰子类中的同名方法，会覆盖父类
		的property属性中的方法

-	`@ParentCls.method.getter/setter/deleter`单独覆盖某个
	property属性方法，但是`ParentCls`是**硬编码方式**，必须
	知道定义property属性的具体类

```python
class Person:
	def __init__(self, name):
		self.name = name

	@property
	def name(self):
		return self._name

	@name.setter
	def name(self, value):
		if not instance(value, str):
			raise TypeError("expected a string")
		self._name = value

	@name.deleter
	def name(self):
		raise AttributeError("can't delete attribute")

class SubPersonAll(Person):
	# 这个类继承、扩展了`name`属性的所有功能
	@property
	def name(self):
		print("getting name")
		return super().name

	@name.setter
	def name(self, value):
		print("Setting name to", value)
		super(SubPerson, SubPerson).name.__set__(self, value)
			# 使用`super(SubPerson, SubPerson)`调用父类实现
			# 将控制权传递给`.name.__set__`方法，委托给父类
				# 中定义的setter方法

	@name.deleter
	def name(self):
		print("deleting name")
		super(SubPerson, SubPerson).name.__delete__(self)

class SubPersonPart(Person):
	# 仅修改`name`属性的某个方法
	# 需要知道定义`name`属性的基类，否则重新定义property属性
		# 的所有方法，并使用`super`将控制权转移给父类
	@Person.name.getter
		# 使用硬编码的`Person`类名，这样会把之前已经定义的
			# property属性方法复制过来，而对应的`getter`、
			# `setter`、`deleter`方法被替换
		# 这里如果直接使用`@property`装饰，那么`setter`、
			# `deleter`方法将会消失
	def name(self):
		print("getting name")
		return super().name
```

###	`__init__`

在基类中写一个公用的`__init__`函数，在子类中省略，避免写很多
重复代码

```python
class Structure:
	_fields = [ ]
	
	def __init__(self, *args, **kwargs):
		if len(args) != len(self._fields):
			raise TypeError("expected {} arguments".format(len(self._fields)))


		for name, value in zip(self._fields, args):
			setattr(self, name, value)
			# 以`_fields`中元素设置属性、值

		extra_args = kwargs.keys() - self._fields
		for name in extra_args:
			setattr(self, name, kwargs.pop(name))
			# 以KEY_WORDS参数设置属性、值

		if kwargs:
			raise TypeError("Depulicate value for {}:".format(",".join(kwargs)))

class Structure2:
	_field = [ ]
	def __init__(self, *args):
		if len(args) != len(self._fields):
			raise TypeError("expected {} arguments".format(len(self._fields)))

		self.__dict__.update(zip(self._fields, args))
			# 直接更新实例`__dict__`
			# 这种方法在子类中设置`__slots__`、通过`@porperty`
				# 包装属性时，不起作用

def test():
	class Stock(Structure):
		_fields = ["name", "share", "price"]

	s1 = Stock("ACME", 50, 91.1)
	s2 = Stock("ACME", 50, 91.1, data="8/2/2012")
```

###	`abc`

定义接口、抽象类，并通过执行类型检查确保子类实现某些方法

-	抽象类无法直接实例化
	-	目的就是让别的类继承它并实现特定的抽象方法
	-	也可以通过注册方式让某类实现抽象基类

-	用途
	-	检查某些类是否为特定类型、实现特定接口

-	尽量`abc`模块作类型检查很方便，但是最好不用过的使用，
	因为python是动态语言，目的就是灵活性，强制类型检查让代码
	更复杂

```python
from abc import ABCMeta, abstractmethod

class IStream(metaclass=ABCMeta):
	@abstractmethod
	def read(self, maxbytes=-1):
		pass

	@abstractmethod
	def write(self, data):
		pass

class SocketStream(IStream):
	# 抽象类目的就是让别的类继承并实现特定抽象方法
	def read(self, maxbytes=-1):
		pass
	def write(self, data):
		pass

def serialize(obj, stream):
	if not isinstance(stream, IStream):
		# 抽象基类的主要用途之一：检查某些类是否为特定类型、
			# 实现特定接口
		raise TypeError("expect an IStream")
	pass

class A(metaclass=ABCMeta):
	@property
	@abstract
	def name(self):
		pass
	
	@name.setter
	@abstractmethod
	def name(self, value):
		pass

	@classmethod
	@abstractmethod
	def method1(cls):
		pass

	@staticmethod
	@abstractmethod
	def method2():
		pass

	# `@abstract`还能注解静态方法、类方法、`@property`
	# 但是需要保证这个方法**紧靠**函数定义
```

标准库中有很多用到抽象基类的地方

-	`collections`模块定义了很多和容器、迭代器（序列、映射、
	集合）有关的抽象基类

	```python
	import collections as clt

	clt.Sequence
	clt.Iterable
	clt.Sized
	clt.Mapping
	```

-	`numbers`库定义了跟数据对象：整数、浮点数、有理数有关的
	基类

-	`IO`库定义了很多跟IO操作相关的基类

###	Mixins

####	继承方案

把一些有用的方法包装成Mixin类，用于扩展其他类的功能，而
这些类往往又没有继承关系

-	Mixin不能直接实例化使用
-	Mixin没有自己的状态信息，即没有定义`__init__`方法，
	没有实例属性，因此Mixin中往往会定义`__slots__ = ()`

```python
class LoggedMappingMixin:
	__slots__ = ()
	def __getitem__(self, key):
		print("getting:", str(key))
		return super9).__getimte__(key)

	def __setitem__(self, key, value):
		print("setting {} = {!r}".format(key, value))
		return super().__setitem__(key, value)

	def __delitem__(self, key):
		print("deleting", str(key))
		return super().__delitem__(key)

class SetOnceMappingMixin:
	__slots__ = ()
	def __setitem__(self, key, value):
		if key in self:
			raise KeyError(str(key), "alreay set")
		return super().__setitem__(key, value)

class StringKeysMappingMixin:
	__slots__ = ()
	def __setitem__(self, key, value):
		if not isinstance(key, str):
			raise TypeError("keys must be strings")
		return super().__setitem__(key, value)

	# 单独的Mixin类使用没有意义，也无法实例化

class LoggedDict(LoggedMappingMixin, dict):
	# 把混入类和其他已存在的类结合起来使用
	pass

from collections import defaultdict

class SetOnceDefaultDict(SetOnceMappingMixin, defaultdict):
	pass

def test():
	d = LoggedDict()
	d["x"] = 23
	print(d["x"])
	del d["x"]

	d = setOnceDefaultDict(list)
	d["x"].append(2)
```

####	类装饰器

```python
def LoggedMapping(cls):
	cls_getitem = cls.__getitem__
	cls_setitem = cls.__setitem__
	cls_delitem = cls.__setitem__
		# 获取原`cls`的方法，避免死循环调用

	def __getitem__(self, key):
		print("getting", str(key))
		return cls_getitem(self, key)
			# 这里使用之前获取的方法指针调用，而不直接使用
				# `cls.__getitem__`避免死循环

	def __setitem__(self, key, value):
		pritn("setting {} = {!r}", str(key))
		return cls_set(self, key, value)

	def __delitem__(self, key):
		print("deleting", str(key))
		return cls_delitem(self, key)

	cls.__getitem__ = __getitem__
	cls.__setitem__ = __setitem__
	cls.__delitem__ = __delitem__

	return cls

@LoggedMapping
class LoggedDict(dict):
	pass
```

##	示例

###	延迟计算

使用描述器类构造延迟计算属性

-	主要目的是为了提升性能
-	避免立即计算
-	缓存计算结果供下次使用

```python
class lazyproperty:
	def __init__(self, func):
		self.func = func

	def __get__(self, instance, cls):
		if instance is None:
			return self
		else:
			value = self.func(instance)
			setattr(instance, self.func.__name__, value)
			# 计算完成之后，缓存计算结果于类实例中
			return value
	# 描述器仅仅定义一个`__get__`方法，比通常具有更弱的绑定
	# 这里，只有被访问属性不在**实例**底层字典`__dict__`中时
		# `__get__`方法才会被触发
	# 描述器属性是类属性，但优先级好像是高于实例属性？？？

def lazyproperty_unchangable(func):
	# 这个版本的延迟计算，使用property属性限制对结果的修改
	name = "_lazy_" + func.__name __

	@property
		# 这里`@property`是在类外定义的
		# 并且，这里实际上返回的是`property`类实例，也不需要
			# `wraps(func)`保持原函数元信息
		# 但此时所有的操作都定向到`getter`函数上，效率较低
	def lazy(self):
		if hasattr(self, name):
			return getattr(self, name)
		else:
			value = func(self)
			setattr(self, name, value)
			return value
	return lazy

import math

class Circle:
	def __init__(self, radiu):
		self.radius = radius

	@lazyproperty
	def area(self):
		print("computing area")
		return math.pi * self.radius ** 2
	# 等价于`area = lazyproperty(area)`
	# 所以是真的把描述器类实例作为类属性

	@lazyproperty
	def perimeter(self):
		print("computing perimeter")
		return 2 * math.pi * self.radius
```

###	数据模型类型约束

使用描述器在对实例某些属性赋值时进行检查

####	类继承方案

#####	基础构建模块

创建数据模型、类型系统的基础构建模块

```python
class Descriptor:
	def __init__(self, name=None, **opts):
		self.name = name
		for key, value in opts.items()
			setattr(self, key, value)

	def __set__(self, instance, value):
		if instance is None:
			return self
		instance.__dict__[self.name] = value

class Typed(Descriptor):
	def __set__(self, instance, value):
		if value < 0:
			raise ValueError("expected >= 0")
		super().__set__(instance, value)

class Unsigned(Descriptor):
	def __set__(self, instance, value):
		if value < 0:
			raise ValueError("expect >= 0")
		super().__set__(instance, value)

class MaxSized(Descriptor):
	def __init__(self, name=None, **opts):
		if "size" not in opts:
			raise TypeError("missing size options")
		super.__init__(name, **opts)

	def __set__(self, instance, value):
		if len(value) >= self.size:
			raise ValueError("size must be <" + str(self.size))
		super().__set__(instance, value)
```

#####	具体数据类型

```python
class Integer(Typed):
	expected_type = int

class UsignedInteger(Integer, Unsigned):
	# 描述器类是基于混入实现的
	pass

class Float(Typed):
	expected_type = Float

class UnsignedFloat(Float, Unsigned):
	pass

class String(Typed):
	expected_type = str

class SizedString(String, MaxSized):
	pass
```

#####	使用

```
class Stock:
	name = SizedString("name", size=8)
	shares = UnsignedInteger("shares")
	price = UnsignedFloat("price")

	def __init__(self, name, shares, price):
		self.name = name
		self.shares = shares
		self.price = price

def check_attributes(**kwargs):
	def decrator(cls):
		for key, value in kwargs.items():
			if isinstance(value, Descriptor):
				value.name = key
				setattr(cls, key, value)
			else:
				setattr(cls, key, value(key))
		return cls
	return decrator

@check_attributes(name=SizedString(size=8),
	shares=UnsignedInteger,
	price=UnsignedFloat)
	# 使用类装饰器简化版本
class Stock2:
	def __init__(self, name, shares, price):
		self.name = name
		self.shares = shares
		self.price = price

class checkmeta(type):
	def __new__(cls, clsname, bases, methods):
		for key, value in method.items():
			if isinstance(value, Descriptor):
				value.name = key
		return type.__new__(cls, clsname, bases, methods)

class Stock3(metaclass=checkdmeta):
	name = SizedString(size=8)
	shares = UnsignedInteger()
	price = UnsignedFloat()

	def __init__(self, name, shares, price):
		self.name = name
		self.shares = shares
		self.price = price
```

####	装饰器类替代mixin

使用类装饰器、元类都可以简化代码，但类装饰器更加灵活

-	类装饰器不依赖其他任何新技术
-	类装饰器可以容易的添加、删除
-	类装饰器能做为mixin的替代技术实现同样的效果，而且速度
	更快，设置一个简单的类型属性值，装饰器要块100%

```python
def Type(expected_type, cls=None):
	if cls is None:
		return lambda cls: Typerd(expected_type, cls)
	super_set = cls.__set__

	def __set__(self, instance, value):
		if no instance(value, expected_type):
			raise TypeError("expect " + str(expected_type))
		super_set(self, instance, value)

	cls.__set__ = __set__
	return cls

def Unsigned(cls):
	super_set = cls.__set__

	def __set__(self, instance, value):
		if value < 0:
			raise TypeError("missing size option")
		super_set(self, name, **opts)

	cls.__init__ = __set__
	return cls

def MaxSized(cls):
	super_init = cls.__init__

	def __init__(self, name=None, **opts):
		if "size" not in opts:
			raise TypeError("missing size option")
		super_init(self, name, **opts)

	cls.__init__ = __init__

	super_set = cls.__set__

	def __set__(self, instance, value):
		if len(value) >= self.size:
			raise ValueError("size must be <" + str(self.size))
		super_set(self, instance, value)

	cls.__set__ = __set__
	return cls

@Typed(int)
class Integer(Descriptor):
	pass

@Unsigned
class UnsignedInteger(Integer):
	pass

@Typed(float)
class Float(Descriptor):
	pass

@Unsigned
class UnsignedFloat(Float):
	pass

@Typed(str)
class String(Descriptor):
	pass

@MaxSized
class SizedString(String):
	pass
```

###	自定义容器

`collections`定义了很多抽象基类，可以用于定义自定义基类

####	`collections.Sequence`

`Sequence`需要实现的抽象方法有：

-	`__getitem__`
-	`__len__`
-	`add`

继承自其的类，支持的常用操作：索引、迭代、包含判断、切片

```python
from collections import Sequence
import collections

class SortedItems(Sequence):
	# 必须要实现所有抽象方法，否则报错
	def __init__(self, initial=None):
		self._items = sorted(initial) if initial is not None else [ ]
	def __getitem__(self, index):
		return self._items[index]

	def __len__(self):
		return len(self._items)

	def add(self, item):
		bisect.insort(self._items, item)
			# `bisect`模块是用于在排序列表中高效插入元素
			# 保证在元素插入之后仍然保持顺序

	# `SortedItems`继承了`colllections.Sequence`，现在和
	# 普通序列无差，支持常用操作：索引、迭代、包含判断、切片

def test():
	items = SortedItems([5, 1, 3])
	print(list(items))
	print(items[0], items[-1])
	items.add(2)
	print(list(items))

	if instance(items, collections.Iterable):
		pass
	if instance(items, collections.Sequence):
		pass
	if instance(items, collections.Container):
		pass
	if instance(items, collections.Sized):
		pass
	if instance(items, collections.Mapping):
		pass
```

####	`collections.MutableSequence`

`MutableSequence`基类包括需要实现的抽象方法

-	`__getitem__`
-	`__setitem__`
-	`__delitem__`
-	`__len__`
-	`insert`

提供的可用方法包括；

-	`append`
-	`count`：统计某值出现的次数
-	`remove`：移除某值的元素

```python
from collections import MutableSequence

class Item(collections.MutableSequence):
	def __init__(self, initial=None):
		self._items = list(initial) if initial is not None else [ ]

	def __getitem__(self, index):
		print("getting:", index)
		return self._items[index]

	def __setitem__(self, index):
		print("setting:", index)
		self._items[index] = value

	def __delitem__(self, index):
		print("deleting:", index)
		del self._items[index]

	def insert(self, index, value):
		print("inserting:", index, value)
		self._items.insert(index, value)

	def __len__(self):
		print("len")
		return len(self._items)

	# 基本支持几乎所有的核心列表方法：`append`、`remove`、
		# `count`

def count():
	a = Items([1, 2, 3])
	print(len(a))
	a.append(4)
		# 在末尾添加元素
		# 调用了`__len__`、`insert`方法
	a.count(2)
		# 统计值为`2`出现的次数
		# 调用`__getitem__`方法
	a.remove(3)
		# 删除值为`3`的元素
		# 调用`__getitem__`、`__delitem__`方法
```

###	属性的代理访问

代理是一种编程模式，将某个操作转移给另一个对象来实现

-	需要代理多个方法时，可以使用`__getattr__`

	-	`__getattr__`方法只在属性、方法不存在时才被调用，
		所以代理类实例本身有该属性不会触发该方法，也不会代理
		至被代理类

	-	如果需要管理所有对方法、属性的访问，可以定义
		`__getattribute__`，其在对类的所有属性、访问时均会
		被触发，且优先级高于`__getattr__`

	-	`__setattr__`、`__delattr__`需要约定是对代理类还是
		被代理类操作，通常约定只代理`_`开头的属性，即代理类
		只暴露被代理类公共属性

	-	注意：对象的元信息直接访问能通过`__getattr__`代理，
		但是对应的hook可能无法正常工作，如果需要，要单独为
		代理类实现**元信息代理方法**

-	通过自定义属性访问方法，可以用不同方式自定义代理类行为，
	如：日志功能、只读访问
-	代理有时候可以作为继承的替代方案：代理类相当于继承了被
	代理类

```python
class Proxy:
	# 这个类用于包装、代理其他类，修改其行为
	def __init__(self, obj):
		self._obj = obj

	def __getattr__(self, name):
		print("getattr:", name)
		return getattr(self._obj, name)

	def __setattr__(self, name, value):
		if name.startwith("_"):
			# 约定只代理不以`_`开头的属性
			# 代理类只暴露被代理类的公共属性
			super().__setattr__(name, value)
		else:
			print("setattr:", name, value)
			setattr(self._obj, name, value)

	def __delattr__(self, name):
		if name.startwith("_"):
			super().__delattr__("name")
		else:
			print("delattr:", name)
			delattr(self._obj, name)

class Spam:
	def __init__(self, x):
		self.x = x

	def bar(self, x):
		print("Spam.bar", self.x, y)

def test():
	s = Spam(2)
	p = Proxy(s)
	p.bar(3)
	p.x = 37
		# 通过`p`代理`s`

	p = Porxy([1, 3, 5])
	# len(p)
		# `len`函数直接使用会报错
	p.__len__()
		# `p.__len__`可以正常代理，返回代理的列表长度
		# 这说明python中的钩子函数有特殊的结构？
```

###	状态机（状态模式）

为不同的状态定义对象，而不是使用过多的条件判断

-	提高执行效率
-	提高代码可维护性、可读性

```python
class Connection:
	def __init__(self):
		self.new_state(ClosedConnectionState)

	def new_state(self, newstate):
		self._state = newstate

	def read(self):
		return self._state.read(self)

	def write(self, data):
		return self._state.write(self, data)

	def open(self):
		return self._state.close(self)

class ConnectionState:
	@staticmethod
	def read(conn):
		raise NotImplementedError()

	@staticmethod
	def write(conn, data):
		raise NotImplementedError()

	@staticmethod
	def open(conn):
		raise NotImplementedError()

	@staticmethod
	def close(conn):
		raise NotImplementedError()

class ClosedConnectionState(ConnectionState):
	@staticmethod
	def read(conn):
		raise RuntimeError("not open")

	@staticmethod
	def write(conn):
		raise RuntimeError("not open")

	@staticmethod
	def open(conn):
		conn.new_state(OpenConnectionState)

	@staticmethod
	def close(conn):
		raise RuntimeError("already closed")

calss OpenConnectionState(ConnectionState):
	@staicmethod
	def read(conn):
		print("reading")

	@staticmethod
	def write(conn, data):
		print("writing", data)

	@staticmethod
	def open(conn):
		raise RuntimeError("already open")

	@staticmethod
	def close(conn):
		conn.new_state(ClosedConnectionState)

def test():
	c = Connection()
	c.open()
	c.read()
	c.write("hello")
	c.close()
```

###	访问者模式

```python
class Node:
	pass

class UnaryOperator(Node):
	def __init__(self, operand):
		self.operand =operand

class BinaryOperator(Node):
	def __init__(self, left, right):
		self.left = left
		self.right = right

class Add(BinaryOperator):
	pass

class Sub(BinaryOperator):
	pass

class Mul(BinaryOperator):
	pass

class Div(BinaryOperator):
	pass

class Nagate(UnaryOperator):
	pass

class Number(Node):
	def __init__(self, value):
		self.value = value

class NodeVsistor:
	def visit(self, node):
		methname = "visit_" + type(node).__name__
		meth = getattr(self, methname, None)
			# 使用`getattr`获取相应方法，避免大量`switch`
			# 子类需要实现`visit_Node`一系列方法
		if meth is None:
			meth = self.generic_visit
		return meth(node)

	def generic_visit(self, node):
		raise RuntimeError("No {} method".format("visit_" + type(node).__name_))

class Evaluator(NodeVisitor):
	def visit_Number(self, node):
		return node.value

	def visit_Add(self, node):
		return self.visit(node.left) + self.visit(node.right)
		# 递归调用`visit`计算结果
		# 因此可能超过python递归嵌套层级限制而失败

	def visit_Sub(self, node):
		return self.visit(node.left) - self.visit(node.right)

	def visit_Mul(self, node):
		return self.visit(node.left) * self.visit(node.right)

	def visit_Div(self, node):
		return self.visit(node.left) / self.visit(node.right)

	def visit_Negate(self, node):
		return -node.operand

def test():
	t1 = Sub(Number(3), Number(4))
	t2 = Mul(Number(2), t1)
	t3 = Div(t2, Number(5))
	t4 = Add(Number(1), t3)

	e = Evaluator()
	e.visit(t4)
```

####	`yield`消除递归

消除递归一般是使用栈、队列，在python还可以使用`yield`得到
更加简洁的代码

```python
import types

class NodeVisitor:
	def visit(self, node):
		stack = [node]
		last_result = None
		while stack:
			try:
				last = stack[-1]
				if isinstance(last, types.GeneratorType):
					# 对`yield`实现
					stack.append(last.send(last_result))
					last_result = None
				elif isinstance(last, Node):
					# 对递归实现
					stack.append(self._visit(stack.pop()))
				else:
					last_result = stack.pop()
			except StopIteration:
				stack.pop()
		return last_result

	def _visit(self, node):
		methname = "visit" + type(node).__name__
		meth = getattr(self, methname, None)
		if meth is None:
			meth = self.generic_visit
		return meth(node)

	def generic_visit(self, node):
		raise RuntimeError("No {} method".format("visit_", type(node).__name__))

class Evaluator(NodeVisitor):
	# `yield`版本不会多次递归，可以接受更多层级
	def visit_Number(self, node):
		return node.value

	def visit_Add(self, node):
		yield (yield node.left) + (yield node.right)
		# 遇到`yield`，生成器返回一个数据并暂时挂起

	def visit_Sub(self, node):
		yield (yield node.left) + (yield node.right)

	def visit_Mul(self, node):
		yield (yield node.left) * (yield node.right)

	def visit_Div(self, node):
		yield (yield node.left) * (yield node.right)

	def visit_Nagate(self, node):
		yield - (yield node.operand)
```

###	字符串调用方法

-	可以使用`getattr(instance, name)`通过字符串调用方法
-	也可以用`operator.methodcaller`
	-	`operator.methodcaller`创建可调用对象，同时提供所有
		必要参数
	-	调用时只需要将实例对象传递给其即可


```python
import math
from operator import methodcaller

class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __repr__(self):
		return "Point({!r}, {!r})".format(self.x, self.y)

	def distance(self, x, y):
		return math.hypot(self.x -x, self.y - y)

def test():
	points = [
		Point(1, 2),
		Point(3, 0),
		Point(10, -3),
		Point(-5, -7)
	]

	points.sort(key=methodcaller("distance', 0, 0))
		# `methodcaller`创建可调用对象，并提供所有必要参数
```

###	缓存实例

####	工厂函数

```python
class Spam:
	def __init__(self, name):
		self.name = name

import weakref
_spam_cache = weakref.WeakValueDictionary()
	# `WeakValueDictionary`实例只保存在其他地方还被使用的
		# 实例，否则从字典移除

def get_spam(name):
	# 使用工厂函数修改实例创建行为
	if name not in _spam_cache:
		s = Spam(name)
		_spam_cache[name] = s
	else:
		s = _spam_cache[name]
	return s

def test():
	a = get_spam("foo")
	b = get_spam("foo")
	print(a is b)
```

####	缓存管理器

将缓存代码放到单独的缓存管理器中，

-	代码更清晰、灵活，可以增加更多的缓存管理机制

```python
import weakref

class CachedSpamManager:
	def __init__(self):
		self._cache = weakref.WeakValueDictionary()

	def get_spam(self, name):
		if name not in self._cache:
			s = Spam(name)
			self._cache[name] = s
		else:
			s = self._cache[name]
		return s

	def clear(self):
		self._cache.clear()

class Spam:
	manager = CacheSpamManager()
	def __init__(self, *args, **kwargs):
		# `__init__`方法抛出异常，防止用户直接初始化
		# 也可以将类名加上`_`，提醒用户不要实例化
		raise RuntimeError("can't instantiate directly")

	@classmethod
	def _new(cls, name):
		self = cls.__new__(cls)
		self.name = name
		return self
```

####	`__new__`

```python
import weakref

clas Spam:
	_spam_cache = weakref.WeakValueDictionary()

	def __new__(cls, name):
		if name in cls._spam_cache:
			return cls._spam_cache[name]
		else:
			self = super().__new__(cls)
			cls._spam_cache[name] = self
			return self

	def __init__(self, name):
		print("initializing Spam")
		self.name = name
	# 这种方式实际会多次调用`__init__`方法，即使已经结果已经
		# 缓存，不是个好方法
```


