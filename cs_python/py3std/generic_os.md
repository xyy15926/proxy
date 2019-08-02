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

