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

