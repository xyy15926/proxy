---
title: 网络、进程间通信
tags:
  - Python
categories:
  - Python
date: 2019-06-10 00:08:54
updated: 2019-06-10 00:08:54
toc: true
mathjax: true
comments: true
description: 网络、进程间通信
---

#	`asyncio`

##	协程与任务

> - <https://docs.python.org/zh-cn/3.10/library/asyncio-task.html>

-	协程包含两层概念
	-	协程函数：定义形式为`async def`的函数
	-	协程对象：调用协程函数返回的对象

-	可等待对象：可在`await`语句种使用的对象
	-	协程对象
	-	`asyncio.Future`：异步调用结果的占位符
		-	以便通过`async/await`使用基于回调的代码
		-	通过情况下无需在应用层级代码中显式创建`Future`
			对象
	-	`asyncio.Task`：`Future`子类，包装coroutine的future

-	运行协程（对象）的三种方式
	-	`await`阻塞式等待协程执行完毕
		-	只能在`async def`函数中使用
		-	`await`同样是在事件循环中阻塞执行
	-	将协程对象包装为可并发运行的`asyncio.Task`，并在事件
		循环中并发执行
		-	`asyncio.create_task`
	-	`asyncio.run()`创建、管理事件循环的高层API
		-	启动事件循环执行是真正运行协程对象的开始

###	`asyncio.run`

```python
def asyncio.run(coro, *, debug=False);
```

-	功能：创建新的事件循环并在结束时关闭
	-	执行传入的协程，并返回结果
	-	管理asyncio事件循环、终结异步生成器、关闭线程池

-	应当被用作asyncio程序的主入口点，理想情况下只被调用一次

> - 同一线程中只能有一个asyncio事件循环运行，若同线程中有
	其他事件循环运行，此函数不能被调用

###	Task

```python
class asyncio.Task(Future):
	def __init__(coro,*,loop=None,name=None);
	# 1、此方法仅在下轮事件循环中`raise asyncio.CancelledError`
	#	给被封包协程，协程自行选择是否取消
	# 2、协程等待的`Future`对象同样会被取消
	def cancel(msg=None);
	bool cancelled();
	bool done();
	def result();
	def exception();
	def add_done_callback(callback, *, context=None);
	def remove_done_callback(callback);
	def get_stack(*, limit=None);
	def print_stack(*, limit=None, file=None);
	def get_coro();
	def get_name();
	def set_name(value);
```

-	`Task`：用于在事件循环中运行协程，非线程安全
	-	若协程在等待`Future`对象，`Task`对象会挂起该协程执行
		并等待该`Future`对象完成再执行
	-	事件循环使用协同日程调度，事件循环每次运行一个`Task`
		对象，`Task`对象会等待`Future`对象完成，事件循环会
		运行其他`Task`、回调、执行IO操作
	-	不建议手动实例化`Task`对象，可以使用高层级的
		`asyncio.create_task()`，或低层级的
		`loop.create_task()`、`ensure_future()`创建

> - `asyncio.Task`从`Future`继承了除`Future.set_result()`、
	`Future.set_exception()`外的所有API

####	`create_task`

```python
def asyncio.create_task(coro, *, name=None);
```

-	功能：将协程打包为task，排入日程准备执行

-	任务会在`get_running_loop()`返回的循环中执行
	-	若线程中没有在运行的循环则引发`RuntimeError`

> - python3.7加入，之前版本可以使用`asyncio.ensure_future()`

####	`gather`

```python
awaitable asyncio.gather(*aws, return_exception=False)
```

-	功能：并发运行`aws`序列中的可等待对象
	-	若`aws`中的某个可等待对象为协程对象，则会自动作为
		任务加入日程
	-	若所有等待对象都成功完成，结果将是所有返回值列表，
		结果顺序同`aws`中对象顺序
	-	若`gather`被取消，被提交的可等待对象也被取消
	-	若`aws`中task、future被取消，将被当作引发
		`CancelledError`处理，`gather`也不会被取消

-	参数说明
	-	`return_exception`
		-	`False`：首个异常被传播给等待`gather()`的任务
		-	`True`：异常和成功结果一样处理并被聚合至结果列表

####	`shield`

```python
awaitable asyncio.shield(aw);
```

-	功能：保护可等待对象防止其被取消
	-	若`aw`是协程，则将自动作为任务加入日程
	-	包含`shield`的协程被取消，`aw`中的任务不会被取消，
		但若`aw`的调用者被取消，`await`表达式仍然会
		`raise CancelledError`
	-	若通过其他方式取消`aw`，则`shield`也会被取消

-	希望完全忽略取消操作则需要配合`try/except`

	```python
	try:
		res = await shield(aw)
	except CancelledError:
		res = None
	```

####	其他

-	Task内省

	```python
	 # 返回当前运行Task实例，没有则返回`None`
	Task = asyncio.current_task(loop=None)
	 # 返回`loop`事件循环未完成Task对象
	Set(Task) = asyncio.current_task(loop=None)
	```

-	Sleep

	```python
	coroutine asyncio.sleep(delay, result=None, *, loop=None)
	```

###	等待超时

####	`wait_for`

```python
coroutine asyncio.wait_for(aw, timeout);
```

-	功能：等待`aw`可等待对象完成
	-	发生超时则取消任务并`raise asyncio.TimeoutError`
	-	函数会等待直到`aw`实际被取消，则总等待时间可能会超过
		`timeout`
	-	可以通过`shield`避免任务被取消
	-	若等待被取消，则`aw`也被取消

####	`wait`

```python
(done, pending) asyncio.wait(aws, *, timeout=None,
	return_when=ALL_COMPELTED);
```

-	功能：并发运行`aws`并阻塞线程直到满足`return_when`指定
	的条件
	-	超时不会`raise asyncio.TimeoutError`，而会在返回未
		完成的`Future`、`Task`

-	参数
	-	`return_when`
		-	`FIRST_COMPLETED`：任意可等待对象结束、取消时
			返回
		-	`ALL_COMPLETED`：所有可等待对象结束、取消时返回
		-	`FIRST_EXCEPTION`：任意可等待对象引发异常结束时
			返回，否则同`ALL_COMPLETED`

####	`as_completed`

```python
iterator asyncio.as_completed(aws, timeout=None);
```

-	功能：并发运行`aws`中可等待对象，返回协程迭代器，返回的
	每个协程可被等待以从剩余可等待对象集合中获得最早下个结果
	-	超时则`raise asyncio.TimeoutError`

```python
for coro in asyncio.as_completed(aws):
	earliest_result = await coro
```

###	其他线程中执行

####	`to_thread`

```python
coroutine asyncio.to_thread(func, *args, **kwargs);
```

-	功能：在不同线程中异步运行函数`func`
	-	`args`、`kwargs`会被直接传给`func`
	-	当前`contextvars.Context`被传播，允许在不同线程中
		访问来自事件循环的上下文变量
	-	主要用于执行可能会阻塞事件循环的函数
		-	对于CPython实现，由于GIL存在，此函数一般只能将
			IO密集型函数变为非阻塞
		-	对于会释放GIL的扩展模块、无此限制的替代性python
			实现，此函数也可以被用于CPU密集型函数

####	`run_coroutine_threadsafe`

```python
concurrent.futures.Future asyncio.run_coroutine_threadsafe(coro, loop)
```

-	功能：向事件循环`loop`提交协程`coro`
	-	线程安全
	-	此函数应该从另一个系统线程中调用

#	`socket`

#	`ssl`

#	`select`

#	`selectors`

#	`asyncore`

#	`asynchat`

#	`signal`

#	`mmap`

