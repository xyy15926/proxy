#	Python运行时服务

##	`sys`

`sys`：与Python解释器**本身相关**的组件

###	平台、版本

```python
import sys
sys.platform
	# 操作系统名称
sys.maxsize
	# 当前计算机最大容纳“原生”整型
	# 一般就是字长
sys.version
	# python解释器版本号
```

####	`sys.xxxcheckinterval`

```python
sys.getcheckinterval()
	# 查看解释器检查线程切换、信号处理器等的频率
sys.setcheckinterval(N)
	# 设置解释器检查线程切换、信号处理器等的频率
```

-	参数
	-	`N`：线程切换前执行指令的数量

-	对大多数程序无需更改此设置，但是可以用于调试线程性能
	-	较大值表示切换频率较低，切换线程开销降低，但是对事件
		的应答能力变弱
	-	较小值表示切换频率较高，切换线程开销增加，对事件应答
		能力提升

####	`sys.hash_info`

```python
sys.hash_info.width
	# `hash()`函数截取hash值长度
```

###	模块搜索路径

```python
sys.path
```

-	返回值：目录名称字符串组成的列表
	-	每个目录名称代表**正在运行**python解释器的运行时模块
		搜索路径
	-	可以类似普通列表在运行时被修改、生效

####	`sys.path`初始化顺序

-	脚本主目录指示器：空字符串

	-	脚本主目录是指脚本**所在目录**，不是`os.getcwd()`
		获取的当前工作目录

-	`PYTHONPATH`环境变量

	```shell
	# .bashrc
	export PYTHONPATH=$PYTHONPATH:/path/to/fold/contains/module
	```

-	标准库目录

-	`.pth`路径文件：在扫描以上目录过程中，遇到`.pth`文件会
	将其中路径加入`sys.path`中

	```conf
	# extras.pth
	/path/to/fold/contains/module
	```

####	导入模块顺序

导入模块时，python解释器

1.	搜索**内置**模块，即内置模块优先级最高
2.	从左至右扫描`sys.path`列表，在列表目录下搜索模块文件

###	嵌入解释器的钩子

```python
sys.modules
	# 已加载模块字典
sys.builtin_module_names
	# 可执行程序的内置模块
sys.getrefcount()
	# 查看对象引用次数
```

###	异常

```python
sys.exc_info()
```

-	返回值：`(type, value, trackback)`
	-	最近异常的类型、值、追踪对象元组

> - 追踪对象可以使用`traceback`模块处理

###	命令行参数

```python
sys.argv
```

-	返回值：命令行参数列表
	-	首项始终为执行脚本名称，交互式python时为空字符串

> - 参数可以自行解析，也可以使用以下标准库中模块
> > -	`getopt`：类似Unix/C同名工具
> > -	`optparse`：功能更加强大

###	标准流

```python
sys.stdin
	# 标准输入流
sys.stdout
	# 标准输出流
sys.stderr
	# 标准错误流
```

-	标准流是**预先打开的python文件对象**

	-	在python启动时自动链接到程序上、绑定至终端
	-	shell会将相应流链接到指定数据源：用户标准输入、文件

####	重定向

-	可以将`sys.stdin`、`sys.stdout`重置到文件类的对象，实现
	python**内部的**、**普遍的**重定向方式

	-	外部：cmd输入输出重定向
	-	局部：指定`print`参数

-	任何方法上与文件类似的对象都可以充当标准流，与对象类型
	无关，只取决于接口

	-	任何提供了类似文件`read`方法的对象可以指定给
		`sys.stdin`，以从该对象`read`读取输入

	-	任何提供了类似文件`write`方法的对象可以指定给
		`sys.write`，将所有标准输出发送至该对象方法上

> - 标准库`io`提供可以用于重定向的类`StringIO`、`ByteIO`
> - 重定向之后`print`、`input`方法将应用在重定向之后的流

####	`stdin`

```python
stdin.read()
	# 从标准输入流引用对象读取数据
input("input a word")
sys.stdin.readlines()[-1]
	# 以上两行语句类似

stdin.isatty()
	# 判断stdin是否连接到终端（是否被重定向）
```

-	在stdin被重定向时，若需要接受用户终端输入，需要使用
	特殊接口从键盘直接读取用户输入
	-	win：`msvcrt`模块
	-	linux：读取`/dev/tty`设备文件

###	退出

```python
sys.exit(N)
```

-	用途：当前**线程**以状态N退出
	-	实际上只是抛出一个内建的`SystemExit`异常，可以被正常
		捕获
	-	等价于显式`raise SystemExit`

> - 进程退出参见`os._exit()`

####	`sys.exitfuncs`

```python
sys.exitfuncs
```

###	编码

```python
sys.getdefaulencoding()
	# 文件内容编码，平台默认值
	# 默认输入解码、输出编码方案
sys.getfilesystemencoding()
	# 文件名编码，平台默认体系
```

> - win10中二者都是`utf-8`，win7中文件名编码是`mbcs`
##	`sysconfig`

##	`builtins`

##	`__main__`

##	`warnings`

##	`dataclass`

##	`atexit`

`atexit`：主要用于在**程序结束前**执行代码

-	类似于析构，主要做资源清理工作

###	`atexit.register`

```python
def register(
	func,
	*arg,
	**kwargs
)
```

-	用途：注册回调函数
	-	在程序退出之前，按照注册顺序反向调用已注册回调函数
	-	如果程序非正常crash、通过`os._exit()`退出，注册回调
		函数不会被调用

```python
import atexit

df func1():
	print("atexit func 1, last out")

def func2(name, age):
	print("atexit func 2")

atexit.register(func1)
atexit.register(func2, "john", 20)

@atexit.register
def func3():
	print("atexit func 3, first out")
```

###	实现

`atexit`内部是通过`sys.exitfunc`实现的

-	将注册函数放到列表中，当程序退出时按照**先进后出**方式
	调用注册的回调函数，

-	若回调函数执行过程中抛出异常，`atexit`捕获异常然后继续
	执行之后回调函数，知道所有回调函数执行完毕再抛出异常

-	二者同时使用，通过`atexit.register`注册回调函数可能不会
	被正常调用

##	`traceback`

###	`traceback.print_tb`

```python
import traceback, sys

try:
	...
except:
	exc_info = sys.exec_info()
	print(exec_info[0], exec_info[1])
	traceback.print_tb(exec_info[2])
```

##	`__future__`

##	`gc`

##	`inspect`

##	`site`

##	`abc`

###	`ABCMeta`

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

###	用途

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

