---
title: Python类用法实例
categories:
  - Python
  - Py3Ref
tags:
  - Python
  - Py3Ref
  - Templates
date: 2019-05-22 08:28:30
updated: 2019-05-22 08:28:30
toc: true
mathjax: true
comments: true
description: Python类用法实例
---

##	延迟计算

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

##	数据模型类型约束

使用描述器在对实例某些属性赋值时进行检查

###	类继承方案

####	基础构建模块

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

####	具体数据类型

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

####	使用

```python
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

###	装饰器类替代mixin

使用类装饰器、元类都可以简化代码，但类装饰器更加灵活

-	类装饰器不依赖其他任何新技术
-	类装饰器可以容易的添加、删除
-	类装饰器能做为mixin的替代技术实现同样的效果，而且速度
	更快，设置一个简单的类型属性值，装饰器要快一倍

####	示例1

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

####	示例2

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

##	自定义容器

`collections`定义了很多抽象基类，可以用于定义自定义基类

###	`collections.Sequence`

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

###	`collections.MutableSequence`

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

##	属性的代理访问

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

##	状态机（状态模式）

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

##	访问者模式

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

###	`yield`消除递归

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

##	字符串调用方法

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

##	缓存实例

###	工厂函数

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

###	缓存管理器

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

###	`__new__`

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


