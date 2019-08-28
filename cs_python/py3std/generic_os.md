---
title: 通用操作系统服务
tags:
  - Python
categories:
  - Python
date: 2019-06-09 23:57:08
updated: 2019-06-09 23:57:08
toc: true
mathjax: true
comments: true
description: 通用操作系统服务
---

##	`os`

`os`：与Python所在**底层操作系统相对应变量、函数**

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

##	`io`

##	`time`

##	`argparse`

`argparse`：编写用户友好的命令行接口

> - <https://docs.python.org/zh-cn/3/library/argparse.html>

###	`argparse.ArgumentParser`

```python
class argparse.ArgumentParser:
	def __init__(self,
		prog=None,				# 程序名，可用`$(prog)s`引用
		usage=None,				# 用法字符串
		description=None,		# 程序描述
		epilog=None,			# 程序尾描述
		parents=[],				# 父解析器，共用当前参数
		formatter_class=argparse.HelpFormatter,
		prefix_chars="-"/str,	# 参数前缀（可包含多个可选）
		fromfile_prefix_chars=None,		# 指定其后参数为存储参数文件名
		argument_default=None,
		conflict_handler="error"/"resolve"	# 默认不允许相同选项字符串
											# 有不同行为，可设置为
											# "resolve"表示允许覆盖
		add_help=True,			# 符解释器中应禁止
		allow_abbrev=True,		# 允许前缀匹配（若不存在重复前缀）
	):
		pass

	# 打印、输出信息至标准输出
	def print_usage(self, file=None/IO) -> None:
		pass
	def print_help(self, file=None/IO) -> None:
		pass
	def format_usage(self) -> str:
		pass
	def format_help(self) -> str:
		pass

	# 退出
	def exit(self, exit=0, message=None):
		pass
	def error(self, message):
		pass
```

####	添加参数

```python
 # 添加参数
def add_argument(self,
	?*name_or_flag=*[str],			# 选项名（无前缀表示位置参数）
	action="store"/"store_const"/	# 参数动作
		"store_true"/"store_false"/
		"append"/"append_const"/
		"count"/"help"/"version",
	nargs=None/int/"?"/"*"/"+"
	const=None,						# `action`、`nargs`所需常数
	default=None,					# 未指定参数默认值
	type=str/type,					# 参数被转换类型、内建函数
	choices=None,					# 参数类型
	required=None,					# 选项候选集
	help=None,						# 选项描述
	metavar=None,					# 参数值示例
	dest=None,						# `parse_args()`返回的参数属性名
):
	pass

 # 添加参数组，参数选项将可以注册至此
def add_argument_group(self,
	title=None,
	description=None
) -> argparse._ArgumentGroup:
	pass

 # 添加互斥参数组，互斥组内仅有一个参数可用
def add_mutally_exclusive_group(self,
	rquired=False				# 互斥组中至少有一个参数
) -> argparse._MutaullyExclusiveGroup:
	pass

 # 设置默认值，优先级高于选项中默认值
def set_defaults(self,
	**kwargs: {参数属性名: 默认值}
):
	pass

 # 获取默认值
def get_default(self,
	**kwargs: {参数属性名: 默认值}
):
	pass
```

-	参数
	-	`action`：关联命令行参数、动作，除以下预定义行为，
		还可以传递`argparse.Action`子类、相同接口类
		-	`store`：存储值，默认行为
		-	`store_const`：存储`const`指定值，通常用于在
			选项中指定标志
		-	`store_true`/`"store_false"`：类上
		-	`append`：存储列表，适合多次使用选项
		-	`append_const`：将`const`参数值追加至列表，
			适合**多个选项需要在同一列表中存储常数**
			（即多个`dest`参数相同）
		-	`count`：计算选项出现次数
		-	`help`：打印完整帮助信息
		-	`version`：打印`version`参数值


	-	`nargs`：参数消耗数目，指定后`pare_args`返回列表，
		否则参数消耗由`action`决定

		-	`int`：消耗参数数目，`nargs=1`产生单元素列表，
			和默认不同
		-	`?/*/+`：类似普通正则，`+`会在没有至少一个参数时
			报错
		-	`argparse.REMAINDER`：所有剩余参数，适合用于从
			命令行传递参数至另一命令行

	-	`const`：保存不从命令行读取、被各种动作需求的常数
		-	`action="store_const"/"append_const"`：必须给出
		-	`nargs=?`：气候选项没有参数时，使用`const`替代

	-	`type`：允许任何类型检查、类型转换，一般内建类型、
		函数可以直接使用
		-	`argparse.FiltType("w")`：为文件读写方便，预定义
			类型转换

	-	`dest`：`parse_args()`返回的参数属性名
		-	位置选项：缺省为首个选项名
		-	关键字选项：优秀首个`--`开头长选项名，选项目中间
			`-`被替换为`_`

####	参数解析

```python
 # 解析参数，无法解析则报错
def parse_args(self,
	args=None/list,			# 需要解析参数列表，默认`sys.argv`
	namespace=None			# 存放属性的`Namespace`对象，缺省创建新空对象
) -> argparse.Namespace:
	pass

 # 解析部分参数，无法解析则返回
def parse_known_args(self,
	args=None/list,
	namespace=None
) -> (argparse.Namespace, [ ]):
	pass

 # 允许混合位置参数、关键字参数
def parse_intermixed_args(self,
	args=None,
	namespace=None
) -> argparse.Namespace:
	pass
def parse_known_intermixed_args(self,
	args=None,
	namespace=None
) -> argparse.Namespace:
	pass
```

-	说明
	-	仅不包含类似负数的选项名，参数才会被尝试解释为负数
		位置参数
	-	前缀无歧义时，可以前缀匹配

####	添加子命令

```python
def add_subparsers(self,
	title,
	description,
	prog,
	parser_class,
	action,
	option_string,
	dest,
	required,
	help,
	metavar
):
	pass
```

###	辅助类

####	动作类

```python
class argparse.Action:
	def __init__(self, option_string, dest, nargs=None, **kwargs):
		pass

	def __call__(self, parser, namespace, values, option_string=None):
		pass
```

####	格式化类

```python
class argparse.ArgumentDefaultHelpFormatter
 # 认为两程序描述已被正确格式化，保留原样输出
class argparse.RawDescriptionHelpFormatter
 # 保留所有种类文字的空格，包括参数描述
class argparse.RawTextHelpFormatter
 # 为每个参数使用`type`参数名作为其显示名，而不是`dest`
class argparse.MetavarTypeHelpFormatter
```

####	其他类

```python
 # 容纳属性的类
class argparse.Namespace:
	pass
 # IO接口简化类
class argparse.FileType:
	def __init__(self,
		mode="r"/"w",
		bufsize=-1,
		encoding=None,
		errors=None
	):
		pass
```

##	`getopt`

##	`optparse`

##	`logging`

##	`logging.config`

##	`logging.handlers`

##	`getpass`

##	`curses`

##	`curses.textpad`

##	`curses.ascii`

##	`curses.panel`

##	`platform`

##	`errno`

##	`ctypes`

