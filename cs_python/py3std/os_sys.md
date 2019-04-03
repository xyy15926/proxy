#	Python系统编程

##	综述

###	主要标准模块

-	`os`：与Python所在**底层操作系统相对应变量、函数**

-	`sys`：与Python解释器**本身相关**的组件

-	文件、目录
	-	`glob`：文件名扩展
	-	`stat`：文件信息

-	并行开发
	-	`threading`、`_thread`、`queue`：运行、同步并发线程
	-	`subprocess`、`multiprocessing`：启动、控制并行进程
	-	`socket`：网络连接、进程间通信

-	系统
	-	`time`、`timeit`：获取系统时间等相关细节
	-	`signal`、`select`、`shutil`、`tempfile`：多种系统
		相关任务

###	说明

-	Python中大部分系统级接口都集中在模块`sys`、`os`中

-	以上模块之间具体实现不总是遵循规则

	-	标准输入、输出流位于`sys`中，但可以将视为与操作系统
		模式相关

-	一些内建函数实际上也是系统接口

	-	`open`

##	`Sys`模块

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
sys.exec_info()
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

##	`os`模块

-	`os`模块提供了*POSIX*工具
	-	**操作系统调用的跨平台移至标准**
	-	不依赖平台的目录处理工具
		-	`os.path`

-	包含在C程序、shell脚本中经常用到的所有操作系统调用，涉及
	目录、进程、shell变量

-	实践中，`os`基本可以作为计算机系统调用的**可移植**接口
	使用
	-	只要技术上可行，`os`模块都能跨平台
	-	但在某些平台，`os`提供专属该平台的工具

###	Shell变量

```python
os.environ
	# 获取、设置shell环境变量，类似字典
os.putenv()
	# 修改进程对应shell环境变量
os.getenv()
```

####	`os.environ`

`os.environ`可以向普通字典一样键索引、赋值

-	默认继承系统所有环境变量、命令行临时环境变量

-	在最新的python中，对`os.environ`的键值修改将自动导出
	到应用的其他部分

	-	`os.environ`对象
	-	进程对应shell环境变量：通过后台调用`os.putenv`生效，
		反之不会更新`os.environ`

-	python进程、链入C模块、该进程派生子进程都可以获取新的
	赋值

	-	子进程一般会继承父进程的环境变量设定
	-	可以作为传递信息的方式

####	`os.putenv`

-	`os.putenv`同时会调用C库中的`putenv`（若在系统中可用）
	导出设置到python链接的C库

	-	底层C库没有`putenv`则可将`os.environ`作为参数传递

###	管理工具

```python
os.getpid()
	# 调用函数的进程id
os.getcwd()
	# 当前工作目录CWD
os.chdir(r"C:\Users")
	# 更改当前工作目录CWD
```

###	移植工具

```python
os.sep
	# python底层运行平台采用的**目录组**分隔符号
	# linux: `/`、win：`\`、某些mac：`:`
os.pathsep
	# 目录列表（字符串形式）中分隔目录的字符
	# posix机：`:`、win：`;`
os.curdir
	# 当前目录代表
	# linux：`.`
os.pardir
	# 父目录代表
	# linux：`..`
os.linesep
	# 换行符
	# linux：`\n`
```

||Linux|Win|Unix|
|------|——————|------|------|
|sep|`/`|`\`|`/`（某些MAC`:`）|
|pathsep|`:`|`;`||
|curdir|`.`|||
|pardir|`..`|||
|linesep|`\n`|`\r\n`||

-	借助这些变量可以系统相关字符串操作的跨平台

> - win下目录组分隔符是`\`，大部分情况下看到`\\`是作为`\`
	转义字符，防止`\`和之后字符转义
> > -	确认不会转义时，直接使用`\`也是可以的
> > -	使用`r''`表示不转义也可以直接使用`\`

###	路径名工具

####	判断存在

```python
os.path.isdir(r"C:\Users")
os.path.isfile(r"C:\Users")
	# 判断路径名是简单文件、目录
os.path.exists(r"C:\Users")
	# 判断路径名是否存在
```

> - `os.stat`配合`stat`模块有更丰富的功能

####	路径操作

```python
pfile = os.path.join(r"C:\temp", "output.txt")
	# 连接文件名、目录

os.path.split(pfile)
	# 分离文件名、目录

os.path.dirname(pfile)
	# 返回路径中目录
os.path.basename(pfile)
	# 返回路径中
os.path.splitext(pfile)
	# 返回文件扩展名

os.path.normpath(r"C:\temp/index.html")
	# 调整路径为当前平台标准，尤其是分隔符混用时
os.path.abspath("index.html")
	# 返回文件的**完整绝对路径名**
	# 扩展`.`、`..`等语法
```

> - `os.sep`配合字符串`.join`、`.split`方法可以实现基本相同
	效果

###	目录、文件操作

```python
os.mkdir(dirname)
os.rename(ori_name, dest_name)
os.remove(filename)
os.unlink(filename)
	# unix下文件删除，同`os.remove`
os.chmod(filename, previlideges)
info = os.stat(filename)
	# 命名元组表示的文件底层信息
	# 可使用`stat`模块处理、解释信息
os.listdir(dirpath)
os.walk(rootdir, topdown=True/False)
	# 遍历根目录下的整个目录树
```

####	`os.listdir`

-	返回值：包含目录中所有条目名称的列表
	-	名称不带目录路径前缀

-	需要注意的是：文件名同样有编码
	-	若参数为字节串，返回文件名列表也是字节串
	-	参数为字符串，返回文件名列表也是字符串

	> - `open`函数也可以类似使用字节串确定需要打开的文件
	> - `glob.glob`，`os.walk`内部都是通过调用`os.listdir`
		实现，行为相同

> - `glob`模块也有遍历目录的能力

####	`os.walk`

-	返回值：返回迭代器
	-	每个元素为`(dirname, subdirs, subfile)`

-	参数
	-	`topdown`：默认`True`，自顶向下返回

###	文件描述符、文件锁

```python
descriptor = os.open(path, flags, mode)
	# 打开文件并返回底层描述符
os.read(descriptor, N)
	# 最多读取N个字节，返回一个字节串
os.write(descriptor, string)
	# 将字节串写入文件
os.lseek(descriptor, position, how)
	# 移动文件游标位置
descriptor.flush()
	# 强制刷出缓冲

new_fd = os.dup(fd)
	# 创建文件描述符副本
os.dup2(fd_src, fd_dest)
	# 将文件描述符`fd_src`复制至`fd_dest`
```

-	`os`通过调用文件的描述符来处理文件

-	基于文件描述符的文件以字节流形式处理

	-	没有字符解码、编码、换行符转换
	-	除了缓冲等额外性能，基于描述符的文件和二进制文件模式
		对象类似

-	文件流对象、工具仅仅是在基于描述符的文件的封装

	-	可以通过`.fileno()`获得文件流对象对应文件描述符，
		`sys.stdin`、`sys.stdout`、`sys.stderr`对应文件
		描述符是：0、1、2

		```python
		os.write(1, b"hello world\n")
		sys.stdout.write("hello world\n")
		```

	-	可以通过`os.fdopen`把文件描述符封装进文件流对象

		```python
		fdfile = os.open("filename", (os.O_RDWR|os.O_BINARY))
		filstream = os.fdopen(fdfile, "r", encoding="utf-8",
			closefd=False)
		```

####	`os.open`

```
def os.open(path,
	flags,
	mode=511, *,
	dir_fd=None
)
```

-	参数
	-	`mode`：需要模式标识符进行二进制操作以得到需要的模式

		-	`os.O_RDWR`
		-	`os.O_RDONLY`
		-	`os.O_WRONLY`
		-	`os.O_BINARY`
		-	`os.O_EXCL`：唯一访问权，是python在并发、进程
			同步情况下锁定文件最便捷的方法
		-	`os.O_NONBLOCK`：非阻塞访问

		> - 其他模式选项参见`os`模块

-	返回值：文件描述符
	-	整数代码、句柄，代表操作系统的中文件

###	退出进程

```c
os._exit(0)
	# 调用进程立即退出，不输出流缓冲、不运行清理处理器
```

##	异常信息

###	`trackback`模块

```python
import traceback, sys

try:
	...
except:
	exc_info = sys.exec_info()
	print(exec_info[0], exec_info[1])
	traceback.print_tb(exec_info[2])
```

##	参数处理

###	`getopt`模块

###	`optparse`模块

##	文件、目录

###	`stat`模块

-	包含`os.stat`信息相关常量、函数以便**跨平台**使用

```python
import stat

info = os.stat(filename)
info[stat.ST_MODE]
	# `stat.ST_MODE`就是字符串
	# 只是这样封装易于跨平台
stat.S_ISDIR(info.st_mode)
	# 通过整数`info.st_mode`判断是否是目录
```

> - `os.path`中包含常用部分相同功能函数

###	`glob`模块

####	`glob.glob`

```python
import glob

def glob.glob(pathname,*,recursive=False)
```

-	参数
	-	`pathname`：文件名模式
		-	接受shell常用文件名模式语法
			-	`?`：单个字符
			-	`*`：任意字符
			-	`[]`：字符选集
		-	`.`开头路径不被以上`?`、`*`匹配
	-	`recursive`
		-	`False`：默认
		-	`True`：`**`将递归匹配所有子目录、文件

-	返回值：匹配文件名列表
	-	目录前缀层次同参数

> - `glob.glob`是利用`glob.fnmatch`模块匹配名称模式

###	`struct`模块

`struct`模块用于打包、解压二进制数据的调用

-	类似于C语言中`struct`声明，需要指定二进制中数据类型
-	可以使用任何一种字节序（大、小端）进行组合、分解

```python
import struct
data = struct.pack(">i4shf", 2, "spam", 3, 1.234)
	# `>`：高位字节优先，大端
	# `i`：整形数据
	# `4s`：4字符字符串
	# `h`：半整数
	# `f`：浮点数
file = open("data.bin", "wb")
file.write(data)
	# 二进制写入字节串
file.close()

file = open("data.bin", "rb")
bytes = file.read()
values = struct.unpack(">i4shf", data)
	# 需要给出字节串存储格式
```

##	系统、信息

###	`locale`模块

```c
import locale

locale.getpreferredencoding()
	# 获取平台默认编码方案
```

###	`dis`模块

```python
def dis.dis(func)
	# 打印可拆解函数语句对应机器指令
```

###	`atexit`模块

`atexit`：主要用于在**程序结束前**执行代码

-	类似于析构，主要做资源清理工作

####	`atexit.register`

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

####	实现

`atexit`内部是通过`sys.exitfunc`实现的

-	将注册函数放到列表中，当程序退出时按照**先进后出**方式
	调用注册的回调函数，

-	若回调函数执行过程中抛出异常，`atexit`捕获异常然后继续
	执行之后回调函数，知道所有回调函数执行完毕再抛出异常

-	二者同时使用，通过`atexit.register`注册回调函数可能不会
	被正常调用

###	`signal`模块

信号模块





