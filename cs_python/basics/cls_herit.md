#	Python类继承

##	访问超类

###	MRO列表

-	对定义的每个类，python会计算一个所谓的方法解析顺序
	（MRO）列表

	-	一个简单的包含所有基类的线性顺序表
	-	可以通过**类**的`__mro__`属性访问

-	MRO列表是通过C3线性化算法实现的，其合并所有父类的MRO
	列表，并遵循如下3条原则

	-	子类先于父类检查
	-	多个父类会根据其在列表中的顺序被检查
	-	如果对下一个类存在多个合法的选择，选择第一个父类

-	为了实现继承，python会在MRO列表上从左到右开始查找基类，
	直到第一个匹配这个属性的类为止

###	`super`

`super`函数可以用于调用父类（超类）方法

-	在`__init__`方法中保证父类正确初始化
-	覆盖python特殊方法
-	尽量使用`super`调用父类方法，而不是直接使用父类名称，
	避免多继承中出现问题，如：超类方法多次调用

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

####	机制

-	调用`super`函数时，python会在MRO表中上继续搜索下个类
	，只要每个重定义的方法统一使用`super`并调用一次，
	控制流会遍历整个MRO列表，每个方法也只会调用一次

	-	所以，在类继承中，尽量使用`super`函数调用超类方法，
		而不要直接使用超类调用方法，否则可能会多次调用同一个
		超类的方法
	-	事实上，`super`并不一定查找到的是某个类在MRO中的下个
		直接父类，甚至可以不是父类

-	因为`super`方法可能调用的不是想要的方法，所以需要遵循
	以下原则

	-	继承体系中，所有相同名字的方法拥有可兼容的参数名，
		比如：相同参数个数、名称
	-	最好确保最顶层类提供这个方法的实现，这样保证MRO上的
		查找链肯定可以找到某个方法

```python
class A:
	def spam(self):
		print("A.spam")
		super().spam()
			# 类`A`没有含有`spam`方法的父类

class B:
	def spam(self):
		print("B.spam")

class C(A, B):
	pass

def test():
	c = C()
	c.spam()
		# `C`从`A`继承`spam`方法
		# `A`中的`spam`方法中的`super`调用了其非父类的类`B`
			# 的`spam`方法
```

##	特殊成员

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

##	抽象类

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

##	Mixins

###	继承方案

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

###	类装饰器

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

