#	类元信息

python对象内`__`开头的方法、属性称为类的元信息

-	对大部分元信息，python有相应的钩子函数调用该方法，用于
	方便访问元信息

-	但是这些钩子函数也不能简单的看作是**直接访问**元信息，
	其可能比直接访问要**更底层**

	-	所以推荐使用钩子函数，而不要直接访问对象元信息
	-	案例可以查看<#属性的代理访问>

-	类继承会获得基类的所有方法
	-	类里面的方法其实真的不是给类使用的，而是给实例使用
	-	类自身使用的方法是元类中的方法

##	实例创建

###	`__new__`

```python
def __new__(*args, **kwargs):
```

-	`__new__`：*built-in function*，创建、返回一个实例

	-	`__new__`接受一个类作为首个参数，返回该类的实例
	-	这里应该就是绑定方法的地方

-	原生有两个`__new__`函数：`object`、`type`

	-	`object.__new__`：给其他类继承，用于创建实例
	-	`type.__new__`：给元类继承，用于创建类


###	`__new__`

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

###	`__init__`

-	**初始化**类实例

	-	这个显然不应该是`cls()`直接调用的函数，因为这个根本
		没有返回值
	-	但这个函数其实泛用性更强，创建实例后被调用、设置实例
		属性，很大程度可以看作是**创建**新实例的函数

###	`__del__`

-	钩子函数：`del`

###	`__prepare__`

在所有类定义开始执行前被调用，用于创建类命名空间，一般这个
方法只是简单的返回一个字典或其他映射对象

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

-	钩子函数：`type`

-	对象类型

###	`__len__`

-	钩子函数：`len`

-	对象长度

##	属性相关

###	`__getattribute__`/`__getattr__`

####	`__getattribute__`

`object.__getattribute__`是`wrapper_descriptor`C实现的函数

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

```python
def __getattribute(self, key):
	"Emulate type_getattro() in Objects/typeobject.c"
	v = object.__getattribute__(self, key)
	if hasattr(v, "__get__"):
		return v.__get__(None, self)
	return v
```

####	`__getattr__`

-	`__getattr__`：仅在`__getattribute__`无法访问属性时调用

	-	是`.`运算符中逻辑，因为直接调用`__getattribute__`
		访问不存在属性不会调用`__getattr__`

	-	`__getattr__`甚至不是`object`具有的
		`<wrapper_descriptor>`

-	相较于`__getattribute__`其实更常用，因为修改**所有**对
	对对象的访问逻辑没啥价值

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

-	钩子函数：`setattr`

###	`__delattr__`

-	钩子函数：`delattr`

##	描述器类

描述器：若类中定义了`__get__`、`__set__`、`__delete__`中至少
一个，则**类的实例**就可以称为描述器

-	描述器可以控制访问属性、方法的行为，是以下的实现机制

	-	`@property`
	-	`super`
	-	静态方法
	-	类方法
	-	属性、实例

-	属性和描述器同名时，在访问“同名属性”时，若为资料描述器则
	优先访问描述器，否则优先访问属性

	> - 资料描述器：同时定义了`__get__`、`__set__`方法
	> - 非资料描述器：只定义了`__get__`方法

	-	可以在`__set__`中`raise AttributeError`得到一个只读
		资料描述器

	-	实例`__dict__`中将没有和资料描述器同名的属性

-	描述器只能定义为类属性，不能定义为实例属性

-	所有对描述器属性的访问会被`__get__`、`__set__`、
	`__delete__`方法捕获到

-	如果只是想简单的自定义某个类的单个属性访问的话，使用
	`@porperty`更容易

###	实现原理

-	描述器的实现依赖于`object.__getattribute__()`方法

	-	可以通过重写类的`__getattribute__`方法改变、关闭
		描述器行为

```python
def __get__(self, obj, type=None)
def __set__(self, obj, value)
def __delete__(self, obj)
```

	# 接受一个实例作为输入，之后相应的操作实例底层字典
	# 这样存储值，虽然描述器类是作为类属性，看起来仍然对
		# 每个实例取不同值

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

###	`__get__`

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

###	`__set__`

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

###	`__delete__`

“删除”描述器类时调用，接受两个参数

```python
def __delete__(self, instance):
	if instance is None:
		pass
	else:
		del instance.__dict__[self.name]
		# 操作实例底层`__dict__`
```

##	字典相关

###	`__getitem__`

###	`__setitem__`

###	`__delitem__`

###	`__hash__`

##	比较相关

###	`__eq__`

-	钩子：`==`/`!=`

-	允许类支持`==`/`!=`运算

###	`__gt__`/`__ge__`

-	钩子：`>`/`>=`

-	允许类支持`>`/`>=`运算

###	`__lt__`/`__le__`

-	钩子：`<`/`<=`

-	允许类支持`<`/`<=`运算

###	`@functools.total_ordering`

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
##	输出相关

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

###	`__str__`

-	钩子函数：`str`

	-	`print`函数对于非`str`类型会隐式调用`str`/`__str__`

-	将实例转换为一个字符串

###	`__repr__`

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

###	`__format__`

-	钩子函数：`format`

-	自定义类的字符串格式

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


##	类属性

###	`cls.__slots___`

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

