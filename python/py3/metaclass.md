#	Python类说明

##	元类说明

###	`type`

就是元类，python中所有类都是type创建

```python
class = type(
	name=cls_name,
	bases=(par_name,..),
	dict={attr1: val1,...}
)
```

-	参数
	-	`cls_name`：类名称
	-	`bases`：父类为元素的元组
	-	`dict`：类方法、属性

-	返回值：类的别名

###	元类作用
	
-	拦截类创建->修改类->返回修改后的类（创建**类对象**）
	（如果不确定是否需要用元类，就不应该使用）

-	元类的作用和函数相似，且事实上，python并不关心**类对象**
	是否是由真正的**元类**创建，可以指定元类为一个函数，而
	不一定是继承自type的元类

-	但是仍然应该尽量将元类指定为继承自type的对象

	-	元类应用场景一般比较复杂，使用类可以更好的管理代码
	-	默认元类是`type`类，类继承保持一致性意图比较明显，且
		可以使用type中的方法
	-	元类可以使用一些类中特有的方法：`__new__`，`__init__`
		等

###	自定义元类

```python

class UpperAttrMeta(type):
	def __new__(cls, cls_name, bases, attrs):
		upper_attrs=dict((name.upper(), val)
			for name,val in attrs.items()
			if not name.startswith('__')
		);
		return super(UpperAttrMeta,cls).__new__(
			cls,
			cls_name,
			bases,
			upper_attrs);
	// 使用元类创建新类

class Foo(metaclass=UpperAttrMeta):
	b=1;
```

使用自定义元类UppAttrMeta创建的类Foo中定义的`__init__`、
`__new__`等函数无意义，因为该类不仅是通过元类创建，也是
通过元类初始化

-	类`Foo`通过`UpperAttrMeta`创建，而`UppAttrMeta`本身没有
	实现自定义`__init__`，默认继承于`object`

	>	因此Foo类的创建就有object的init完成
		segmentfault.com/q/1010000004438156
		这个是原话，不明白什么意思了

-	但是如果元类仅仅是`pass`，如下：
	```python
	class MetaCls(type):
		pass;
	```
	使用此自定义元类，类定义中的`__init__`、`__new__`有效
		

####	类创建


###	py2自定义元类

python创建**类对象**步骤

-	`__metaclass__`指定创建类使用的元类

	-	按照优先级：*类定义内 > 祖先类内 > 模块内 > `type`*，
		查找`__metaclass__`，并使用其**创建类对象**，若前三者
		均未定义`__metaclass__`，则使用`type`创建

	-	自定义元类就是为`__metaclass__`指定自定义值

	-	python只是将创建类的参数传递给`__metaclass__`，并不
		关心`__metaclass__`是否是一个**类**
		-	`cls()`返回一个类对象，是相当于调用`cls.__new__`
		-	所以可以指定`__metaclass__`为一个函数

```python
		
def upper_attr(cls_name, bases, attrs):
	upper_attrs=dict((name.upper(), val) for name,val in attrs.items());
	return type(cls_name, bases, upper_attrs);

class Foo():
	bar=1;
	__metaclass__=upper_attr;

	# 函数版本

class UpperAttrMeta(type):
	def __new__(clsf, cls_name, bases, attrs):
		upper_attrs=dict((name.upper(), val) for name,val in attrs.items());
		return type(cls_name, bases, upper_attrs); 

class Foo():
	bar=1;
	__metaclass__=UpperAttrMeta;

	# 类版本1

class UpperAttrMeta(type):
	def __new__(cls, cls_name, bases, attrs):
		upper_attrs=dict((name.upper(), val) for name,val in attrs.items());
		return type.__new__(cls, cls_name, bases, upper_attrs); 

	# 类版本2

class UpperAttrMeta(type):
	def __new__(cls, cls_name, bases, attrs):
		upper_attrs=dict((name.upper(), val) for name,val in attrs.items());
		return super(UpperAttrMeta,cls).__new__(cls, cls_name, bases, upper_attrs); 

	# 类版本3
```


##	元类示例

###	缓存实例

```python
import weakref

class Cached(type):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__cache = weakref.WeakValueDictionary()

	def __call__(self, *args):
		if args in self.__cache:
			return self.__cache[args]
		else:
			obj = super().__call__(*args)
			self.__cache[args] = obj
			return obj

class Spam(metaclass=Cached):
	def __init__(self, name):
		print("Creating Spam({!r})".format(name))
		self.name = name
```

###	捕获类属性定义顺序

```python
from collection import OrderedDict

class Typed:
	_expected_type = type(None)
	def __init__(self, name=None):
		self._name = name

	def __set__(self, instance, value):
		if not instance(value, self_expected_type):
			raise TypeError("expected" + str(self._expected_type))
		instance.__dict__[self._name] = value

class Integer(Typed):
	_expected_type = int

class Float(Typed):
	_expected_type = float

class String(Typed):
	_expected_type = str

class OrderedMate(type):
	def __new__(cls, clsname, bases, clsdict):
		d = dict(clsdict)
		order = [ ]
		for name, value in clsdict.items():
			if isinstance(value, Typed):
				value._name = name
				order.append(name)
		d["_order"] = order
		return type.__new__(cls, clsname, bases, d)

	@classmethod
	def __prepare__(cls, clsname, bases):
		# 此方法会在开始定义类、其父类时执行
		# 必须返回一个映射对象，以便在类定义体中使用
		return OrderedDict()

class Structure(metaclass=OrderedMeta):
	def as_csv(self):
		return ",".join(str(getattr(self, name)) for name in self._order)

class Stock(Structure):
	name = String()
	shares = Integer()
	price = Float()

	def __init__(self, name, shares, price):
		self.name = name
		self.shares = shares
		self.price = price
```

###	有可选参数元类

为了使元类支持关键字参数，必须在`__prepare__`、`__new__`、
`__init__`方法中使用KEYWORDS_ONLY关键字参数 

```python
class MyMeta(type):
	@classmethod
	def __prepare__(cls, name, bases, *, debug=False, synchronize=False):
		pass
		return super().__prepare(naeme, bases)

	def __new__(cls, name, bases, *, debug=False, synchronize=False):
		pass
		return super().__new__(cls, name, bases, ns)

	def __init__(self, name, bases, ns, *, debug=False, synchronize=False):
		pass
		super().__init__(name, base, ns)
```

