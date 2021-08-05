---
title: Py3std Readme
categories:
  - Python
  - Py3std
tags:
  - Python
  - Py3std
  - Readme
date: 2019-04-20 15:39:09
updated: 2019-04-20 15:39:09
toc: true
mathjax: true
comments: true
description: Py3std说明
---

##	常用参数说明

-	函数书写声明同Python全局说明
-	以下常用参数如不特殊注明，按照此解释

###	Common

###	Stream


-	`mode="r"/"w"/"a"/"+"/"t"/"b"`
	-	含义：文件/管道打开模式
		-	`t`：文本，可省略
		-	`b`：二进制
		-	`r`：读，默认
		-	`w`：写
		-	`a`：追加，大部分支持
		-	`+`：更新模式，同时允许读写
			-	`r+`：文件已存在，读、写
			-	`w+`：清除之前内容，读、写
			-	`a+`：读、追加写
	-	默认：`rt`/`r`

-	`buffering/bufsize = -1/0/1/int`
	-	含义：缓冲模式
		-	`0`：不缓冲，只在二进制模式中被运行
		-	`1`：逐行缓冲，只在文本模式中有效
		-	其他正整数：指定固定大小chunk缓冲的大小
		-	`-1`：全缓冲
			-	普通二进制、文本，缓冲chunks大小启发式确定，
				`io.DEFAULT_BUFFER_SIZE`查询
			-	终端交互流（`.isatty()`），逐行缓冲
	-	默认：`-1`

-	`encoding(str)`
	-	含义：文件编码
		-	`utf-8`
		-	`utf-16`
		-	`utf-16-le`
		-	`utf-16-be`
		-	`utf-32`
		-	`gbxxxx`
		-	待续
	-	缺省：使用`locale.getpreferedencoding()`返回值

###	Threading/Processing

-	`block/blocking = True/False`

	-	含义：是否阻塞
	-	默认：大部分为`True`（阻塞）
	-	其他
		-	对返回值不是bool类型的函数，非阻塞时若无法进行
			操作，往往会`raise Exception`

-	`timeout = None/num`

	-	含义：延迟时间，单位一般是秒
	-	默认：None，无限时间
	-	其他
		-	`block=False`时，一般`timeout`参数设置无效

-	`fn/func/callable(callable)`

	-	含义：可调用对象
	-	默认：一般默认值
	-	其他
		-	实参可以是任何可调用对象
			-	函数
			-	方法
			-	可调用对象

-	`args = ()/None/tuple/list`/`*args(arg_1, ...)`

	-	含义：函数位置参数
	-	默认：`()/None`，无参数

-	`kwrags/kwds = {}/None/dict`/`**kwargs(kwarg_1=v1, ...)`

	-	含义：函数关键字参数
	-	默认：`{}/None`，无参数

-	`callback=callable`

	-	含义：回调函数
		-	异步线程、进程调用才会有该参数
		-	回调函数接收进程/线程返回值作为参数
		-	回调函数最好有返回值，否则会阻塞进程、线程池
	-	默认：`None`，无参数

-	`chunksize=None/1/int`
	-	含义：一次传递给**子进程**的迭代器元素数量
		-	常在进程池迭代调度函数中，较大的`chunksize`
			能减少进程间通信消耗，但会降低灵活性
		-	线程调度相关函数该参数被忽略
	-	默认：`None`/`1`，一次传递一个元素

-	`daemon=False/None/True`
	-	含义：是否为守护进程/线程
		-	默认情况下，主进程（线程）会等待子进程、线程退出
			后退出
		-	主进程（线程）不等待守护进程、线程退出后再退出
		-	注意：主进程退出之前，守护进程、线程会自动终止

##	Python命令行参数

-	`-c`：解释执行语句
-	`-u`：强制输入、输出流无缓冲，直接输入，默认全缓冲



