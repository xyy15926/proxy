---
title: 数据模型--函数决定对象
categories:
  - Python
  - Py3Ref
tags:
  - Python
  - Py3Ref
  - Data Model
  - Function
  - Coroutine
  - Asynchronous
date: 2019-06-10 01:36:53
updated: 2022-06-27 10:34:50
toc: true
mathjax: true
comments: true
description: 数据模型--函数决定对象
---

##	*Generator*

-	*Generator Functions* 生成器函数：包含 `yield` 表达式的函数（方法）
	-	生成器函数调用时返回生成器迭代器对象，控制执行函数体

-	*Generator-Iterator* 生成器（迭代器）：生成器函数执行得到的迭代器
	-	生成器迭代器可视为利用 `yield` 表达式对迭代器的的简化实现
		-	实现有迭代器协议 `__iter__`、`__next__`
		-	迭代过程由生成器函数负责，无需手动维护重入状态
	-	同时，生成器迭代器功能有扩展
		-	额外定义有 `.send`、`.throw`、`.close` 方法，用以与生成器迭代器交互
		-	定义有状态概念，可通过 `gi_running` 属性判断是否在执行

> - <https://docs.python.org/zh-cn/3.9/reference/expressions.html#yield-expressions>

###	*Generator-Iterator*

-	生成器迭代器工作过程
	-	首次尝试迭代（`__next__`、`send`） 时，生成器函数开始执行
	-	执行到 `yield` 表达式被挂起，保留所有局部状态，然后返回 `expression_list`
		-	局部变量当前绑定
		-	指令指针
		-	内部求值栈
		-	任何异常处理状态
	-	继续迭代（`__next__`、`send`），生成器函数继续执行，回到上步
	-	执行至 `return` 语句，生成器函数体执行完毕，将 `raise StopIteration`
		-	异常 `value` 属性被置为返回值

-	*GI* 在声明周期中有如下4中状态
	-	`"GEN_CREATED"`：等待执行
	-	`"GEN_RUNNING"`：正在执行，只有多线程中才能看到
	-	`"GEN_SUSPENDED"`：在yield表达式处挂起状态
	-	`"GEN_CLOSED"`：关闭状态

> - *GI* 状态可通过`inspect.getgeneratorstate()`方法查看

####	`try`

-	`try` 结构中任何位置都允许 `yield` 表达式，因此注意
	-	若 *GI* 在销毁之前未恢复执行，`try` 结构中的 `yield` 表达式挂起可能导致 `finally` 子句执行失败
	-	对异步生成器，应由运行该异步生成器的事件循环、或任务调度器负责调用 `close()` 方法，从而允许任何挂起的 `finally` 子句得以执行

####	`yield from`

-	`yield from` 将控制流转移给其后迭代器
	-	即，父生成器将部分操作委托给另一迭代器
		-	允许生成器将包含 `yield` 表达式部分代码分离至另一生成器
		-	且，生成器无需额外设置值传递、异常捕获逻辑，而通过 `yield from` 全权委托
	-	提供了主调者、子生成器之间的双向通道
		-	主调者、子生成器之间可以双向传递值、异常
		-	否则需通过迭代、`.send`、`.throw` 手动处理
		-	且，自动处理 `StopIteration`，将其 `value` 属性值（即返回值）返回给父生成器

> - `yield` 相应的是将控制流转移给主调者

###	*GI* 特殊方法

-	`GI.__next__`：开始生成器函数执行、或从上次执行 `yield` 表达式处恢复执行
	-	编译器预定义函数：无法传递参数
		-	返回值：*GI* 产生的下个值，即 `yield` 表达式中 `expression_list` 值
		-	若生成器未产生下个值就退出，则 `raise StopIteration`
	-	说明
		-	*GI* 通过 `__next__` 方法恢复执行时，`yield` 表达式始终返回 `None`
		-	执行至下个 `yield` 表达式
		-	此方法通常隐式通过 `for` 循环、`next()`函数调用

-	`GI.send(value)`：向其“发送”值 `value`，再恢复生成器函数执行并
	-	编译器预定义函数
		-	`value`：作为 *GI* 中 `yield` 表达式结果
		-	返回值：*GI* 产生的下个值（同 `GI.__next__`）
	-	说明
		-	若 *GI* 中包含子迭代器，`send` 传参数值将被传递给下层迭代器
			-	若子迭代器没有合适接收方法、处理，将 `raise AttributeError`、`raise TypeError`
			-	但 `.send(None)` 也能正常工作
			-	可理解为 `.send` 方法参数为 `None` 时实际不会有传递参数行为
		-	调用 `.send` 启动 *GI* 时，必须以 `None` 作为调用参数
			-	解释器执行至 `YIELD_VALUE` （即对应 `yield` 表达式）时即挂起，则启动 *GI* 时无法处理传入值
			-	恢复 *GI* 执行时，则以传入值作为 `yields` 表达式返回值，从上次挂起处开始执行

-	`GI.throw(value)`：在生成器函数暂停处引发 `type` 异常
	-	编译器预定义函数
		-	`value`：异常类
		-	返回值：返回生成器产生的下个值（若异常被处理，类似 `send`）
	-	说明
		-	若异常被处理：则执行直至下个 `yield` 表达式
		-	若未捕获传入异常、或引发另一个异常：则异常会被传播给调用者
		-	若生成器中包含子迭代器，`throw` 传入异常将被传递给子迭代器
		-	调用 `.throw` 启动生成器，会在生成器函数开头引发错误

-	`GI.close()`：在生成器函数暂停处 `raise GeneratorExit`
	-	编译器定义函数
	-	说明
		-	若之后生成器函数正常退出、引发 `GeneratorExit` （或未捕获）：则关闭生成器，并忽略该异常
			-	即 `close` 忽略 `GeneratorExit`
			-	关闭的生成器继续迭代将 `raise StopIteration`
		-	若生成器函数继续 `yield` 值（即生成器函数捕获异常后）：则 `raise RuntimeError`
		-	若生成器引发其他异常：则传播给调用者
		-	若生成器已异常、或正常而退出：则无操作

> - *GI* 已经执行时，调用上述任何方法将 `raise ValueError`
> - `GeneratorExit` 不衍生自 `Exception`，二者均衍生自 `BaseException`

###	类比：调用函数、协程

-	`yield` 表达式类似 **调用外部函数**
	-	当前函数栈保留当前状态、挂起
	-	执行 `yield` 表达式“完毕”后，“返回”到调用处
	-	`yield` 表达式“返回值”取决于生成器恢复执行所调用方法
		-	`.__next__`：`for`、`next()` 调用，返回 `None`
		-	`.send()`：返回 `.send()` 参数
	-	此时整体类似于
		-	主调函数调用生成器函数
		-	生成器函数通过 `yield` 表达式 “调用主调函数”

-	因此，生成器函数类似协程
	-	也被称为 *semi-coroutine*（半协程）
		-	可 `yield` 多次，有多个入口点
		-	执行可以被挂起
		-	可在恢复执行时传递参数控制执行
	-	但 `yield` 表达式仅能传递值给主调者，并不是指定需要跳转的、平等的协程
		-	`yield` 后控制权总是转移给生成器迭代器调用者
		-	生成器函数无法控制 `yield` 后继续执行的位置
	-	而 `yield from` 带来控制权的自由转移，则设计适当生成器，即基本可实现协程
		-	Python3.5 以前即通过 `asyncio.coroutine`、`yield from` 实现事件循环的协程模型
		-	因此，新式协程函数中 `yield from` 关键字被禁止

##	*Coroutine*

###	*Awaitable*

-	*Awaitable* 可等待对象：异步调用句柄
	-	在 `await` 语句中作为目标被等待
	-	可等待对象通常为实现 `__await__` 方法对象，主要包括 3 种
		-	协程
		-	任务
		-	`asyncio.Future` 对象

> - `types.coroutine`、`asyncio.coroutine` 装饰的生成器返回的 *GI* 未实现 `__await__`，是旧时的基于生成的生成器的协程

####	`await`、`__await__`

-	`obj.__await__(self)`：返回供控制权转移的迭代器
	-	普通方法
		-	返回值：必须为迭代器，否则 `raise TypeError`
	-	钩子：`await`
	-	说明
		-	返回的迭代器即用于控制权转移的对象
			-	迭代器迭代值必须为 `None`，否则 `raise RuntimeError`
		-	迭代器返回值即可等待对象返回的结果
			-	实际上存储在 `StopIteration.value` 中
			-	`await` 会自动处理异常、获取 `value`

-	`await` 即自动迭代 `__await__` 迭代器
	-	可视为不断自动向 `__await__` 迭代器 `send(None)`
	-	自动处理 `StopIteration`，并将其 `value` 属性作为返回值
		-	基本类似 `yield from OBJ.__await__()`
		-	故可等待对象不应直接 `raise StopIteration`
	-	`await` 协程一次以上将 `raise RuntimeError`

###	*Coroutine*

-	*Coroutine Function* 协程函数：`async def` 定义的函数
	-	协程函数调用时返回协程对象，控制执行函数体
		-	可以视为返回待执行函数体
	-	`await` 表达式、`async with` 语句、`async for` 语句只能在协程函数体中使用

-	*Coroutine Objects* 协程对象：调用协程函数返回的 *awaitable* 对象
	-	协程执行：迭代 `__await__` 返回的迭代器以控制协程执行
		-	协程执行、重入即通过迭代 `__await__` 迭代器实现控制权转移而达成
		-	协程结束执行、返回时，迭代器将 `raise StopIteration`，并将异常的 `value` 属性置为返回值
			-	`await` 会自动处理异常，并获取 `value` 属性返回
			-	手动调用 `.send()` 、处理异常亦可
		-	协程引发异常会（通过迭代器）传播
	-	协程函数结果总为协程对象，即使其中不包含任何 `await`
		-	或者，协程对象 `__await__` 总返回迭代器，即使协程函数体中无额外重入点

> - 协程能多个不同点上进入、挂起、恢复，是子例程的更一般形式
> - <https://docs.python.org/zh-cn/3.9/library/asyncio-task.html>
> - <https://docs.python.org/zh-cn/3.9/reference/datamodel.html#coroutines>
> - <https://docs.python.org/zh-cn/3.9/reference/compound_stmts.html#coroutines>

###	协程特殊方法

```python
def iter_N(N: int=3):
	for i in range(N):
		print((yield))				# 被委托生成器即可打印协程 `.send` 传入值
	return N
class Awaitable:
	def __await__(self):
		return iter_N()
async def gen_co():
	ret = await Awaitable()			# 包含 `yield` 将变为异步生成器，故利用可等待对象
	return ret
```

-	`coroutine.send(value)`：开始、恢复协程执行（类似生成器 `send`）
	-	协程对象编译器预定义方法
		-	`value`：发送给迭代器（当前导致协程挂起的迭代器）值
			-	取 `None` 时同等待 `__await__()` 迭代器迭代下一项
		-	返回值、异常类似于通过 `send` 迭代 `__await__()` 迭代器
	-	说明
		-	类似 `yield from`，此方法被委托给导致协程挂起的迭代器的 `send()` 方法

-	`coroutine.throw(value)`：在协程内引发指定异常（类似生成器 `throw`）
	-	协程对象编译器预定义方法
		-	`value`：发送给迭代器（当前导致协程挂起的迭代器）的异常
		-	返回值、异常类似于通过 `throw` 迭代 `__await__()` 迭代器
	-	说明
		-	类似 `yield from`，此方法被委托给导致协程挂起的迭代器的 `throw()` 方法
		-	将在协程挂起处 `raise` 指定异常
		-	若异常未在协程内被捕获，则传回给主调者

-	`coroutine.close()`：清理协程对象自身并退出（类似生成器 `close`）
	-	说明
		-	类似 `yield from`，此方法被委托给导致协程挂起的迭代器的 `close()` 方法
			-	在协程挂起处 `raise GeneratorExit`，使得协程立即清理自身
			-	协程被标记为已结束执行，即使未被启动
		-	协程对象将被销毁时，会被自动调用

-	示例重要说明
	-	为能让主调者获取控制权以手动调用 `send`（且不结束），需要触发 `yield`
	-	为规避 `yield` 带来的异步生成器转换，故通过自定义 `__await__` 返回生成器
	-	实务中，可等待对象的 `__await__` 迭代器长度无意义（自动迭代完），关键是控制权的流转
	-	即仅考虑 `await` 的等待、控制权流转
		-	忽略 `__await__` 迭代器
		-	忽略 `send`、`throw`、`close` 等方法

##	*Asynchronous Generator*

-	*Asynchronous Generator Functions* 异步生成器函数：包含 `yield` 语句、使用 `async def` 定义函数、方法（即在协程中）
	-	返回异步生成器迭代器对象，控制执行函数体

-	*Asynchronous Generator-Iterator* 异步生成器（迭代器）：异步生成器函数返回的异步迭代器
	-	调用异步生成器 `asend`、`__anext__` 等方法将返回可等待对象
		-	多个可等待对象执行同一函数栈
		-	执行（一般即 `await`）可等待对象才开始执行函数体，至 `yield` 表达式处挂起
			-	异步迭代器迭代不会引发异常（即使被关闭），仅在执行时会引发相应异常（包括 `Async StopIteration`）
		-	迭代出的可等待对象含义只是为：执行函数体至 `yield` 挂起
			-	先后迭代出的可等待对象不存在依赖关系
			-	任何迭代出的可等待对象执行结果只取决于异步生成器状态，与迭代位次无关
	-	类似于（同步）生成器
		-	可使用 `async for` 进行异步迭代
		-	可通过 `asend` 方法向 `yield` 表达式传值
		-	挂起即局部状态被保留，等恢复执行后继续执行
		-	`try` 结构中可能因 `yield` 表达式挂起而执行 `finally` 子句失败
		-	异步生成器函数中异常可通过异步迭代器返回的可等待对象执行而抛出

> - <https://docs.python.org/zh-cn/3.9/reference/expressions.html?highlight=aclose#asynchronous-generator-functions>
> - 异步生成器函数中 `yield from` 将引发预发错误

###	异步生成器特殊方法

-	`agen.__anext__(self)`：返回可等待对象，执行后返回下个迭代值
	-	异步生成器编译器预定义方法
		-	返回值：代表执行异步生成器函数体的可等待对象
	-	说明
		-	运行（`await`）`__anext__` 可等待对象时，类似生成器 `__next__`
			-	启动、从挂起处开始执行异步生成器函数，直至下个`yield`
			-	`yield` 表达式取值 `None`
			-	若异步生成器退出，将 `raise StopAsyncIteration`
		-	运行可等待对象返回值
	-	钩子：`async for` 即隐式调用此方法

> - `__anext__` 是异步迭代器协议内容，任何异步迭代器均应实现此方法

-	`agen.asend(value)`：返回可等待对象，执行后传递 `value`、返回下个迭代值
	-	异步生成器编译器预定义方法
		-	返回值：代表执行异步生成器函数体的可等待对象
	-	说明
		-	运行（`await`）`__asend__` 可等待对象时，类似生成器 `send`
			-	启动（必须传递 `None`）、从挂起处开始执行异步生成器函数，直至下个`yield`
			-	`yield` 表达式取值 `value`
			-	若异步生成器退出，将 `raise StopAsyncIteration`
		-	运行可等待对象返回值

-	`agen.athrow(value)`：返回可等待对象，执行后在挂起处抛出异常、返回下个迭代值（若异常被处理）
	-	异步生成器编译器预定义方法
		-	返回值：代表执行异步生成器函数体的可等待对象
	-	说明
		-	运行（`await`）`__athrow__` 可等待对象时，类似生成器 `throw`
			-	在挂起处引发异常，若通过 `athrow` 启动异步生成器，则直接在开头处抛出异常
			-	若异常被处理，则执行至下个 `yield` 并返回迭代值
			-	若异常未被处理、或引发其他异常，异常被传播给调用者
			-	若异步生成器退出，将 `raise StopAsyncIteration`
		-	运行可等待对象返回值

-	`agen.aclose()`：返回可等待对象，执行后在挂起处 `raise GeneratorExit`
	-	异步生成器编译器预定义方法
		-	返回值：代表执行异步生成器函数体的可等待对象
	-	说明
		-	若之后异步生成器函数正常退出、引发 `GeneratorExit` （或未捕获）：则关闭生成器，并忽略该异常
			-	即 `close` 忽略 `GeneratorExit`
			-	执行的可等待对象所属异步生成器被关闭则将 `raise StopAsyncIteration`
		-	若异步生成器函数继续 `yield`（即生成器函数捕获异常后）：则 `raise RuntimeError`
		-	若异步生成器引发其他异常：则传播给调用者
		-	若异步生成器已异常、或正常而退出：则无操作

##	类异步功能

###	异步迭代协议

-	`async.__aiter__(self)`：创建、返回异步迭代器对象
	-	普通方法
		-	返回值：异步迭代器对象，返回其他任何对象都将 `raise TypeError`
	-	说明：类似 `__iter__` 方法
		-	仅负责创建、返回实现 `__anext__` 的迭代器，不负责产生、返回迭代器元素
		-	异步迭代器对象也需实现此方法（返回自身），以配合 `async for` 语句
	-	钩子：`async for`
		-	`async for` 将默认调用对象 `__aiter__` 方法获取异步迭代器对象

-	`async.__anext__(self)`：返回可等待对象，执行后从异步迭代器返回下个结果值
	-	普通协程方法
		-	返回值：可等待对象
	-	典型实现
		-	应使用 `async def` 定义为协程，其中包含异步代码
	-	说明：类似 `__next__` 方法
		-	迭代结束后，执行返回的可等待对象应 `raise StopAsyncIteration`，且后续调用需始终如此
	-	钩子：`async for`
		-	此方法（异步迭代器）常用于 `async for` 语句中自动迭代

###	异步上下文管理器协议

-	`async.__aenter__(self)`：异步创建、进入关联当前对象的上下文执行环境
	-	普通协程方法
		-	返回：可等待对象
	-	典型实现
		-	由 `async def` 定义为协程函数，即在创建上下文执行环境时可以被挂起
	-	钩子：`async with`
		-	异步上下文管理器常用于 `async with` 中自动创建、销毁执行环境

-	`async.__aexit__(self)`：异步销毁、退出关联当前对象的上下文执行环境
	-	普通协程方法
		-	返回：可等待对象
	-	典型实现
		-	由 `async def` 定义为协程函数，即在销毁上下文执行环境时可以被挂起
	-	钩子：`async with`
		-	异步上下文管理器常用于 `async with` 中自动创建、销毁执行环境


