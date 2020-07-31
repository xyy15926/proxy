---
title: Python元编程
tags:
  - Python
categories:
  - Python
date: 2019-03-30 01:54:46
updated: 2019-02-17 11:57:07
toc: true
mathjax: true
comments: true
description: Python元编程
---

##	Decorator装饰器

装饰器就是函数，接受函数作为**参数**并返回**新的函数**

-	装饰器不会修改原始函数签名、返回值，但是大部分返回的
	新函数不是原始函数，看起来像是函数元信息发生改变
-	**新函数**也可能是单纯的修改原函数的元信息、然后返回

###	装饰器设计

####	保留函数元信息

装饰器作用在函数上时，**原函数**的重要元信息会丢失

-	名字：`func.__name__`
-	文档字符串：`func.__doc__`
-	注解：
-	参数签名：`func.__annotations__`

```python
import time
from functools import wraps

def timethis(func):
	@wraps(func)
	def wrapper(*args, **kargs):
		start = time.time()
		result = func(*args, **kwargs)
		end = time.time()
		print(func.__name__, end - start)
		return result
	return wrapper

@timethis
def countdown(n):
	while n > 0:
		n -= 1

def countdown(n):
	pass
countdown = timethis(countdown)
	# 这个和上面装饰器写法效果一致
```

-	`@wraps`能够复制原始函数的元信息，并赋给装饰器返回的函数
	，即被`@wraps`装饰的函数

-	装饰后函数拥有`__wrapped__`属性
	-	直接用于访问被包装函数，即解除装饰器
		-	有多个包装器`__wrapped__`的行为是不可预知的，
			可能会因为python版本有差，应该避免
	-	让装饰器函数正确暴露底层参数签名信息

	```python
	countdown.__wrapped__(10000)
	from inspect import signature
	print(signature(countdown))
	```

####	自定义属性

在装饰器中引入**访问函数**，访问函数中使用`nolocal`修改内部
变量t

-	访问函数可在多层装饰器间传播，如果所有的装饰中的wrapper
	都使用了`@functools.wraps`注解（装饰）
-	可以使用`lambda`匿名函数，修改访问函数属性改变其行为

```python
from functools import wraps
import logging

def logged(level, name=None, message=None):
	def decorator(func):
		logname = name if name else func.__module__
		log = logging.getLogger(logname)
		logmsg = message if message else func.__name__

		@wraps(func)
		def wrapper(*args, **kwargs):
			log.log(level, logmsg)
			return func(*args, **kwargs)

		@attach_wrapper(wrapper)
			# attach setter function
		def set_level(newlevel):
			nonlocal level
			level = newlevel

		@attach_wrapper(wrapper)
		def set_message(newmsg):
			nonlocal logmsg
			logmsg = newmsg

		return wrapper
	return decorator

@logged(logging.DEBUG)
def add(x, y):
	return x + y

@logged(logging.CRITICAL, "example")
def spam():
	print("Spam")

@timethis
@logged(logging.DEBUG)
	# 使用多层装饰器，访问函数可以在多层装饰器间传播
def countdown(n):
	while n > 0:
		n -= 1
```

###	带参装饰器

带参装饰器就是接受参数、处理，再返回一个**第一个参数**为函数
的函数（内部装饰器）

####	三层函数

-	**最外层**套一层**接受**参数的函数
-	内部装饰器函数可以访问这些参数，并在“存储”在内部，
	相当于一个闭包
-	返回使用参数处理后的**装饰器函数**，再装饰函数

```python
from functools import wraps
import logging

def logged(level, name=None, message=None):

	r"""Decorator that allows logging
	:param level: logging level
	:param name: logger name, default function's module
	:param message: log message, default function's name
	:return :decorator
	"""

	def decorator(func):
		logname = name if name else func.__module__
		log = logging.getLogger(logname)
		logmsg = message if message else func.__name__

		@wraps(func)
		def wrapper(*args, **kwargs):
			log.log(level, logmsg)
			return func(*args, **kwargs)
		return wrapper

	return decorator

@logged(logging.DEBUG)
def add(x, y):
	return x + y

@logged(loggin.CRITICAL, "example")
def spam():
	print("spam")

def spam():
	pass
spam = logged(logging.CRITICAL, "example")(spam)
	# 这样调用和之前的装饰器语句效果相同
```

####	`functools.partial`

`partial`接受一个函数作参数，并返回设置了部分参数默认值的
函数，而最外层函数就只是用于“获取”参数，因此可以使用此技巧
减少一层函数嵌套

```python
from functools import partial

def attach_wrapper(obj, func=None):

	r"""Decorator to attach function to obj as an attr
	:param obj: wapper to be attached to
	:param func: function to be attached as attr
	:return: wrapper
	"""

	if func is None:
		return partial(attach_wrapper, obj)
	setattr(obj, func.__name__, func)
	return func

```

####	参数可选

#####	三层函数

这种形式的带可选参数的装饰器，即使不传递参数，也必须使用
调用形式装饰

```python
from functools import wraps
import logging

def logged(level=logging.DEBUG, name=None, message=None):

	r"""Decorator that allows logging
	:param level: logging level
	:param name: logger name, default function's module
	:param message: log message, default function's name
	:return :decorator
	"""

	def decorator(func):
		logname = name if name else func.__module__
		log = logging.getLogger(logname)
		logmsg = message if message else func.__name__

		@wraps(func)
		def wrapper(*args, **kwargs):
			log.log(level, logmsg)
			return func(*args, **kwargs)
		return wrapper

	return decorator

@logged()
	# 这种方式实现的默认参数装饰器必须使用`@logged()`调用的
	# 形式，从装饰器形式就可以看出，必须调用一次才能返回内部
	# 装饰器
	# 这种形式的装饰器不符合用户习惯，不用传参也必须使用的
	# 调用形式
def add(x, y):
	return x + y

@logged(level=logging.CRITICAL, name="example")
def add(x, y):
	return x + y
```

#####	`partial`

这种形式的装饰器，不传参时可以像无参数装饰器一样使用

```python
from functools import wraps, partial
import logging

def logged(
	func=None,
	*,
	level=logging.DEBUG,
	name=None,
	message=None):
	if func is None:
		return partial(logged, level=level, name=name, message=message)
	logname = name if name else func.__module__
	log = logging.getLogger(logname)
	logmsg = message if message else func.__name__

	@wraps(func)
	def wrapper(*args, **kwargs):
		log.log(level, logmsg)
		return func(*args, **kwargs)

	return wrapper

@logged
	# 使用`partial`函数形式的带默认参数的装饰器，可以不用
	# 调用形式
def add(x, y):
	return x + y

@logged(level=logging.CRITICAL, name="example")
def spam():
	print("spam")
```

###	用途示例

####	强制检查参数类型

```python
def typeasssert(*ty_args, **ty_kwargs):

    r"""Decorator that assert the parameters type
    :param *ty_args: parameters type indicated with position
    :param **ty_kwargs: parameters type indicated with keywords
    :return: wrapper
    """

    def decorator(func):
        # if in optimized mode, disable type checking
        if not __debug__:
            return func

        sig = signature(func)
        # map function argument names to asserted types
        bound_types = sig.bind_partial(*ty_args, **ty_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            # map function argument names to paraments
            bound_values = sig.bind(*args, **kwargs).argument
            for name, val in bound_values.items():
                if name in bound_types:
                    if not isinstance(value, bound_types[name]):
                        raise TypeError(
                            "Argument {} must be {}.".format(name, bound_types[name])

            return func(*args, **kwargs)
        return wrapper
    return decorator
```

####	装饰器类

为了定义类装饰器，类需要实现`__call__`、`__get__`方法，然后
就可以当作普通的的装饰器函数使用

```python
import types
from functools improt wraps

class Profiled:
	def __init__(self, func):
		wraps(func)(self)
			# 获取`func`的元信息赋给实例
		self.ncalls = 0

	def __call__(self, *args, **kwargs):
		self.nacalls += 1
		return self.__wrapped__(*args, **kwargs)
			# 解除装饰器

	def __get__(self, instance, cls):
		if intance is None:
			return self
		else:
			return types.MethodType(self, instance)
```

####	装饰器方法

在类中定义装饰器方法，可以将多个装饰器关联于同一个类的实例，
方便在装饰器中记录、绑定、共享信息

-	`@property`装饰器类：可以将类方法`method_atrr`“转变”为
	属性
	-	设置完成之后，会创建2个以方法名开头的装饰器
		`method_attr.setter`、`method_attr.deleter`用于装饰
		**同名**的方法
	-	分别对应其中包含`setter`、`deleter`、`getter`
		（`@property`自身）三个方法

	>	详情查看`clsobj`

####	装饰类、静态方法

装饰器必须放在`@staticmethod`、`@classmethod`装饰器之前
（内层），因为这两个装饰器实际上并不创建callable对象，而是
创建特殊的描述器对象

####	添加参数

为原函数“添加”KEYWORD_ONLY参数，这种做法不常见，有时能避免
重复代码

-	KEYWORD_ONLY参数容易被添加进`*args`、`**kwargs`中
-	KEYWORD_ONLY会被作为特殊情况挑选出来，并且不会用于调用
	原函数
-	但是需要注意，被添加的函数名称不能与原函数冲突

```python
from functools import wraps
import inspect
def optional_debug(func):
	if "debug" in inspect.getargspect(func).args:
		raise TypeError("Debug argument already defined")
		# 防止原函数参数中参数与新增`debug`冲突

	@wraps(func)
	def wrapper(*args, debug=False, **kwargs):
		# 给原函数“增加”了参数
		if debug:
			print("calling", func.__name__)
		return func(*args, **kwargs)
	return wrapper
```

####	装饰器扩充类功能

可以通过装饰器修改类某个方法、属性，修改其行为

-	可以作为其他高级技术：mixin、metaclass的简洁替代方案
-	更加直观
-	不会引入新的继承体系
-	不依赖`super`函数，速度更快

注意：和装饰函数不同，装饰类不会返回新的类，而是修改原类，
因此装饰器的顺序尤为重要

```python
def log_getattribute(cls):
	orig_getattribute = cls.__getattribute__

	def new_getattribute(self, name):
		print("getting:", name)
		return orig_getattribute(self, name)

	cls.__getattribute = new_getattribute
	return cls

@log_getattribute
class A:
	def __init__(self, x):
		self.x = x
	def spam(self):
		pass
```

##	Metaclass元类


###	用途示例

####	单例模式

```python
class Singleton(type):
	def __init__(self, *args, **kwargs):
		self.__intance = None
		super().__init__(*args, **kwargs)

	def __call__(self, *args, **kwargs):
		if self.__instance is None:
			self.__instance = super().__call__(*args, **kwargs)
			return self.__instance
		else:
			return self.__instance

class Spam(metaclass=Singleton):
	def __init__(self):
		print("Creating Spam")
```

####	缓存实例

```python
import weakref
class Cached(type):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__cache = weakref.WeakValueDictionar()

	def __call__(self, *args):
		if args in self.__cache:
			return self.__cache[args]
		else:
			obj = super().__call__(*args)
			self.__cache[args] = obj
			return obj

class Spam(metaclass=Cached):
	def __init__(self, name):
		print("creating spam({!r})".format(name))
		self.name = name
```
