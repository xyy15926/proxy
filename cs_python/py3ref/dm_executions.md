---
title: 数据模型--执行相关
tags:
  - Python
  - Py3Ref
categories:
  - Python
  - Py3Ref
date: 2019-05-20 22:24:33
updated: 2019-05-20 22:24:33
toc: true
mathjax: true
comments: true
description: 数据模型--执行相关
---

-	以下类型为内部类型，由解释器内部使用、但被暴露给用户，
	其定义可能随着未来解释器版本更新而变化

	-	代码对象
	-	帧对象
	-	回溯对象
	-	切片对象：参见*cs_python/py3ref/dm_basics*
	-	静态方法对象：参见*cs_python/py3ref/#todo*
	-	类方法对象：参见*cs_python/py3ref/#todo*

##	*Module*

模块：python代码的基本组织单元

-	导入系统创建
	-	`import`语句
	-	`importlibd.import_module()`、`__import__()`函数
-	模块对象具有由字典`__dict__`实现的命名空间
	-	属性引用：被转换为该字典中查找`m.__dict__['x']`
	-	属性赋值：更新模块命名字典空间
	-	不包含用于初始化模块的代码对象
	-	模块中定义函数`__globals__`属性引用其

###	元属性

-	`__name__`：模块名称
-	`__doc__`：模块文档字符串
-	`__annotaion__`：包含变量标注的字典
	-	在模块体执行时获取
-	`__file__`：模块对应的被加载文件的路径名
	-	若加载自一个文件，某些类型模块可能没有
		-	C模块静态链接至解释器内部
	-	从共享库动态加载的扩展模块，该属性为共享库文件路径名
-	`__dict__`：以字典对象表示的模块命名空间

> - CPython：由于CPython清理模块字典的设定，模块离开作用域时
	模块字典将被清理，即使字典还有活动引用，可以复制该字典、
	保持模块状态以直接使用其字典

##	*Code Object*

代码对象：“伪编译”为字节的可执行python代码，也称*bytecode*

-	代码对象和函数对象区别
	-	代码对象不包含上下文；函数对象包含对函数全局对象
		（函数所属模块）的显式引用
	-	默认参数值存放于函数对象而不是代码对象
	-	代码对象不可变，也不包含对可变对象的应用

-	代码对象由内置`compile()`函数返回
	-	可以通过函数对象`__code__`属性从中提取
	-	可以作为参数传给`exec()`、`eval()`函数执行

###	特殊属性

-	`co_name`：函数名称
-	`co_argcount`：位置参数数量
-	`co_nlocals`：函数使用的本地变量数量（包括参数）
-	`co_varnames`：包含本地变量名称的元组
-	`co_freevars`：包含自由变量的元组
-	`co_code`：表示字节码指令序列的字符串
-	`co_consts`：包含字节码所使用的字面值元组
	-	若代码对象表示一个函数，第一项为函数文档字符，没有
		则为`None`
-	`co_names`：包含字节码所使用的名称的元组
-	`co_filenames`：被编译代码所在文件名
-	`co_firstlineno`：函数首行行号
-	`co_lnotab`：以编码表示的字节码偏移量到行号映射的字符串
-	`co_stacksize`：要求栈大小（包括本地变量）
-	`co_flags`：以编码表示的多个解释器所用标志的整形数
	-	`0x04`位：函数使用`*arguments`接受任意数量位置参数
	-	`0x08`位：函数使用`**keywords`接受任意数量关键字参数
	-	`0x20`位：函数是生成器
	-	`0x2000`位：函数编译时使用启用未来除法特性
	-	其他位被保留为内部使用

##	*Frame Objects*

栈帧对象：执行帧

-	可能出现在回溯对象中，还会被传递给注册跟踪函数

###	特殊只读属性

-	`f_back`：前一帧对象，指向主调函数
	-	最底层堆栈帧则为`None`
-	`f_code`：此栈帧中所执行的代码对象
-	`f_locals`：查找本地变量的字典
-	`f_globals`：查找全局变量
-	`f_builtins`：查找内置名称
-	`f_lasti`：精确指令，代码对象字节码字符串的索引

###	特殊可写属性

-	`f_trace`：`None`，或代码执行期间调用各类事件的函数
	-	通常每行新源码触发一个事件
-	`f_trace_lines`：设置是否每行新源码触发一个事件
-	`f_trace_opcodes`：设置是否允许按操作码请求事件
-	`f_lineno`：帧当前行号
	-	可以通过写入`f_lineno`实现Jump命令

###	方法

-	`.clear()`：清楚该帧持有的全部对本地变量的引用
	-	若该栈帧为属于生成器，生成器被完成
	-	有助于打破包含帧对象的循环引用
	-	若帧当前正在执行则会`raise RuntimeError`

##	*Traceback Objects*

回溯对象：表示异常的栈跟踪记录

-	异常被印发时会自动创建回溯对象，并将其关联到异常的可写
	`__traceback__`属性

	-	查找异常句柄使得执行栈展开时，会在每个展开层级的当前
		回溯之前插入回溯对象
	-	进入异常句柄时，栈跟踪将对程序启用
	-	获取：`sys.exc_info()`返回的元组第三项、异常的
		`__traceback__`属性
	-	程序没有合适的处理句柄时，栈跟踪将写入标准错误

-	可通过`types.TracebackType`显式创建
	-	由回溯对象创建者决定如何链接`tb_next`属性构成完整
		栈追踪

###	特殊只读属性

-	`tb_frame`：执行当前层级的执行栈帧
-	`tb_lineno`：给出发生异常所在行号
-	`tb_lasti`：最后具体指令

> - 若异常出现在没有匹配的`except`子句、没有`finally`子句
	的`try`中，回溯对象中的行号、最后指令可能于相应帧对象中
	行号不同

###	特殊可写属性

-	`tb_next`：栈跟踪中下一层级（通往发生异常的帧），没有
	下一层级则为`None`

##	I/O对象/文件对象

文件对象：表示打开的文件

-	创建文件对象
	-	`open()`内置函数
	-	`os.popen()`、`os.fdopen()`
	-	`socket.makefile()`

-	`sys.stdin`、`sys.stdout`、`sys.stderr`会初始化为对应于
	解释器的标准输入、输出、错误流对象
	-	均以文本模式打开
	-	遵循`io.TextIOBase`抽象类所定义接口

