#	Python类、对象

##	类元信息

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

	def __format__(self):
		if self.x == 0:
			return self.y
		elif self.y == 0:
			return self.x
		return "{0.x!r}, {0.y!r}".format(self)
```

####	`__str__`

-	将实例转换为一个字符串
-	内置`str`调用实例的`__repr__`方法，即`print`的输出

####	`__repr__`

-	返回实例的代码表示形式
	-	其生成的文本字符串标准的做法是需要让
		`eval(repr(x))==x`为真，至少也需要创建有用的文本
		表示，并使用`<>`扩起
	-	通常用于重新构造实例

-	内置`repr`调用实例`__repr__`方法，也即交互环境下直接
	“执行”变量的结果

-	格式化代码中`!r`指明输出使用`__repr__`而不是默认
	`__str___`
	-	如果`__str__`没有定义，会使用`__repr__`代替输出

####	`__format__`

-	自定义类的字符串格式
-	内置`format`函数调用类的`__format__`方法，如果
	`__format__`方法有参数，`format`也可以传递更多参数

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
-	不论`with`代码块中发生什么，即使是出现异常，`__exit__`
	控制流也会执行完
-	方法的第三个参数`exc_val`包含异常类型、异常值、和
	回溯信息，可以自定义方法决定如何利用异常信息
	-	返回`True`， 异常会被清空
-	使用`__enter__`、`__exit__`、`with`自动控制资源释放，
	有效避免死锁

####	`cls.__slots___`

-	定义类属性`__slots__`后，python会为实例属性使用紧凑内部
	表示
	-	实例属性使用固定大小、很小的数组构建，而不是为每个
		实例定义字典
	-	在`__slots__`列出的属性名在内部映射到数组指定小标上
-	因此不能再给实例添加新的属性，只能使用在`__slots__`中
定义的属性名
-	类似于R中`factor`类型、C中`enum`类型

```python
class Date:
	__slots__ = ["year", "month", "day"]
	def __init__(self, year, month, day):
		self.year = year
		self.month = month
		self.day = day
```
