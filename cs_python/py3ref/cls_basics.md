---
title: 类
tags:
  - Python
  - Py3Ref
categories:
  - Python
  - Py3Ref
date: 2019-05-25 19:53:54
updated: 2019-05-25 19:53:54
toc: true
mathjax: true
comments: true
description: 类
---

##	综述

###	*Custom Classes*

用户定义类：通过类定义创建

-	每个类通过字典对象`__dict__`实现独立的命名空间
	-	类属性引用被转化为在此字典中查找
	-	其中未发现属性名时，继续在基类中查找
		-	基类查找使用C3方法解析顺序，即MRO列表
	-	也存在一些**钩子对象**允许其他定位属性的方式

-	当类属性引用*yield*类方法对象时，其将转化为`__self__`
	属性为当前类对象的**实例方法对象**

-	当类属性引用*yield*静态方法对象时，其将转换为静态方法
	对象所封装的对象

-	类属性复制会更新类字典，不会更新基类字典

-	类对象可被调用产生类实例

####	特殊属性

-	`__bases__`：包含基类的元组，依在基类列表中出现的顺序

###	*Class Instances*

类实例：通过**调用类对象**创建

-	每个类实例都有一个通过字典对象`__dict__`实现的独立命名
	空间
	-	属性引用首先在此字典中查找
	-	其中未发现属性名时，继续在对应类属性中查找
		-	用户定义函数对象：其会被转化为**实例方法对象**
			-	`__self__`属性即为该实例
		-	静态方法、类方法对象：同样会被转化

		> - 描述器属性有特殊处理，实际存放在类`__dict__`中
			对象不同
	-	若未找到类属性，对象对应类具有`__getattr__()`方法，
		将调用该方法

-	属性赋值、删除会更新实例字典，不会更新对应类字典
	-	若类具有`__setattr__`、`__delattr__`方法，将调用方法
		而不是直接更更新对应实例字典

####	特殊属性

-	`__class__`：实例对应类

###	*Classes*

类：类对象通常作为“工厂”创建自身实例

-	`__doc__`：类的文档字符串
	-	类定义第一条语句、且须为字符串字面值
	-	没有则为`None`
	-	不会被子类继承

###	*Class Instances*

类实例：在所属类中定义`__call__()`方法即成为可调用对象

##	属性

###	属性访问`.`

-	`A.attr`被解释为`type(A)中__getattribute__(A, attr)`
	-	`.`的行为由python解释器定义
	-	`type(A)中__getattribute__`的**中**用于强调不会再从
		`type(type(A))`继续获取调用`__getattibute__`

-	则**定义在类命名空间中函数是为实例定义**
	-	要为类定义方法应该自定义元类

-	测试代码

	```python
	class Meta(type):
		def __getattribute__(self, attr):
			print("--Meta--", attr)
			return super().attr

	class D(metaclass=Meta):
		def __getattribute__(self, attr):
			print("--Class--", attr)
			return super().attr
	```

> - `__getattribute__`函数说明参见
	*cs_python/py3ref/special_methods*

###	属性访问控制

> - python没有没有属性访问控制，不依赖语言特性封装数据，而是
	遵循一定属性、方法命名规约达到效果

-	`__`开头属性：属性名称会被修改
	-	防止被派生类继承，此类属性无法通过继承覆盖
	-	即若清楚代码会涉及子类，且应该在子类中隐藏起来，考虑
		使用双下划线开头

	> - 通常是在属性名称前添加类名标记`_cls`
	> - 但同时以`__`结尾属性名称不会被修改

-	单`_`开头属性：应被视为私有属性，不应被外部访问
	-	python无法真正防止访问内部名称，但是这样会导致脆弱的
		代码
	-	此约定同样适用于模块名、模块级别函数
		-	默认情况下，通配符`*`不会导入模块私有属性，除非
			在配置有`__all__`属性

		> - 导入参见*cs_python/py3ref/simple_stmt#todo*

-	单`_`结尾：避免定义的变量和保留关键字冲突

###	特殊属性

-	`__dict__`：命名空间包含的属性
-	`__doc__`：文档字符串
	-	第一条语句、且须为字符串字面值
	-	没有则为`None`
	-	不会被子类继承
-	`__name__`：名称
-	`__qualname__`：*qualified name*，完整限定名称
	-	以点号分隔的名称
	-	显示模块全局作用域到模块中某个定义类、函数、方法的
		路径
-	`__module__`：所属模块名称
	-	没有则为`None`

##	描述器属性

> - 描述器协议参见*cs_python/py3ref/special_methods*
> - 实例/类/静态方法：参见*cs_python/py3ref/dm_gfuncs*

###	`@property`

`@property`装饰器：为类的属性增加处理逻辑，如：类型检查、
合法性验证

-	*property*属性和普通属性实现迥异，但使用类似
	-	*property*属性就是绑定有这些处理逻辑函数的类实例
	-	访问、赋值、解除绑定时会自动触发`getter`、`setter`、
		`deleter`处理逻辑

-	*property*属性（或者说有效描述器）为类属性
	-	一般需要通过在实例、或描述器命名空间
		`instance.__dict__`中存储数据，以实现对实例操作逻辑
		独立
	-	也可以实时计算属性值，此时无需为实例分别存储数据
	-	初始化时，不应该直接设置底层数据属性，会绕过`setter`
		的参数检查

> - 过度使用`@property`时会降低代码可读性、效率，使用
	*get/set*方法可能有更好的兼容性

####	代码实现

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

> - `@property`是描述器类，接受方法返回同名资料描述器

####	创建*property*属性

-	`@property[.getter]`装饰*getter-like*方法得到同名资料
	描述器

-	返回描述器包含`.setter()`、`.deleter()`方法/装饰器进一步
	完善描述器

	-	`@method.setter`：为描述器完善赋值处理逻辑
	-	`@method.deleter`：为描述器完善`del`处理逻辑

-	可以直接使用已有类中函数创建`property`类实例，得到
	*property*属性（描述器）

-	派生类中*property*属性覆盖

	-	派生类中直接使用`@property`创建同名属性会覆盖基类
		中*property*属性
		-	只有显式声明的处理逻辑被设置
		-	基类中逻辑位于基类相应同名*property*属性，不会
			被“隐式继承”

	-	`@<basecls>.<method>.getter/setter/deleter`单独覆盖
		*property*属性方法
		-	但是`basecls`是**硬编码方式**，必须知道定义
			property属性的具体类（或其子类）

> - 描述器协议、实现参见*cs_python/py3ref/special_methods*

####	示例

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

##	类继承

-	类继承会获得基类的所有方法
	-	类里面的方法其实真的不是给类使用的，而是给实例使用
	-	类自身使用的方法是元类中的方法

###	*Method Resolution Order*

*MRO*/方法解析顺序列表：包含当前类所有超类的线性顺序表

-	MRO列表顺序通过C3线性化算法实现，对每个类按以下规则合并
	所有父类的MRO列表

	-	子类先于父类检查
	-	多个父类根据其在列表中的顺序被检查
	-	若对下一个类存在多个合法的选择，选择第一个父类

-	为了实现继承，python会在MRO列表上从左到右开始查找超类，
	直到第一个匹配这个属性的类为止

> - 可以通过**类**`__mro__`、`mro()`访问

###	`super`

```python
class super:
	super()
		# 等同于：`super(__class__, <first_argument>)`
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
	-	第一个参数：在MRO列表中定位类搜索起点（不包括）
	-	第二个参数：**提供MRO列表**
		-	类：直接传递MRO列表
		-	实例：传递所属类的MRO列表

-	返回：封装有两个参数的`super`实例
	-	类似于返回MRO列表中某个类的实例，取决于访问的属性

-	用途：依次遍历MRO列表（指定位置开始）中类，查找指定属性
	-	可以使用指定超类创建`super`实例，跳过对部分类搜索
	-	只有MRO列表中每个类中的方法都`super()`调用，才能保证
		列表中所有类的该方法都被链式调用

####	说明

-	需要注意`super(cls, inst).__getattribute__("meth")`中
	共有两段属性访问，两次访问调用不同`__getattribute__`

	-	`super(cls, inst).__getattribute__`首先调用
		`super.__getattribute__`在`type(inst).mro()`中寻找
		`some_cls.__getattribute__`

	-	然后调用`some_cls.__getattrbibute__("meth")`访问
		`meth`属性

-	应使用`super`访问基类属性，而不是直接使用基类名称，避免
	多继承中出现问题

	-	继承链`super`保证方法只按找MRO列表顺序调用一次
	-	多继承中硬编码基类名称调用方法可能导致方法被调用多次

-	`super`访问的属性路线不够明确，所以需要遵循以下原则

	-	继承体系中，所有相同名字的方法拥有可兼容的参数名，
		比如：相同参数个数、名称
	-	最好确保最顶层类提供这个方法的实现，这样保证MRO上的
		查找链肯定可以找到该方法

###	抽象类

接口、抽象类

-	抽象类无法直接实例化
	-	目的就是让别的类继承它并实现特定的抽象方法
	-	也可以通过注册方式让某类实现抽象基类

-	用途
	-	通过执行类型检查，确保实现为特定类型、实现特定接口

> - 类型检查很方便，但是不应该过度使用，因为动态语言目的就是
	灵活性，强制类型检查让代码更复杂
> - 使用`abc`模块方便定义抽象类

###	Mixins

*Mixins*：把一些有用的方法包装成Mixin类，用于扩展其他类的
功能，而这些类往往又没有继承关系

-	Mixin不能直接实例化使用
-	Mixin没有自己的状态信息，即没有定义`__init__`方法，
	没有实例属性，因此Mixin中往往会定义`__slots__ = ()`

> - Mixins讨论参见*cs_program/program_design/inheritation*

####	例

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

