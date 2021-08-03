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
updated: 2021-08-02 11:46:05
toc: true
mathjax: true
comments: true
description: 数据模型--函数决定对象
---

##	函数

###	用户定义函数

用户定义函数：通过函数定义创建，调用时附带参数列表

-	函数对象支持获取、设置任意属性
	-	用于给函数附加元数据
	-	使用属性点号`.`获取、设置此类属性

####	特殊属性

-	`__defaults__`：有默认值参数的默认值组成元组
	-	没有具有默认值参数则为`None`
-	`__code__`：**编译后**函数体代码对象
-	`__globals__`：存放函数中全局变量的字典的引用
	-	即引用函数所属模块的全局命名空间
	-	只读
-	`__closure__`：包含函数**自由变量**绑定单元的元组
	-	没有则为`None`
	-	只读
-	`__annotations__`：包含参数注释的字典
	-	字典键为参数名、`return`（若包含返回值）
	-	将变量名称（非私有）映射到标注值的特殊字典
	-	若该属性可写、在类或模块体开始执行时，若静态地发现
		标注则被自动创建
-	`__kwdefaults__`：`keyword-only`参数的默认值字典

> - 大部分可写属性会检查赋值类型
> - 自由变量：上层命名空间中变量，不包括顶级命名空间，即
	全局变量不是自由变量

###	实例/绑定方法

实例/绑定方法：使用**属性表示法调用**、**定义在类命名空间**
中的函数

-	实例方法用于将类、类实例同可调用对象结合起来
	-	可调用对象：**须为类属性，即定义在类命名空间中**，
		通常为用户定义函数、类方法对象

-	通过实例访问类命名空间定义函数时**创建**实例/绑定方法
	-	通过示例、类访问的命名空间定义函数类型不同
		-	`wrapper_descriptor`转换为`method-wrapper`
		-	`function`转换为`method`
		-	通过类访问方法，得到是普通函数
	-	绑定：此时会将`self`作为**首个参数**添加到参数列表
	-	调用实例方法时，调用相应下层函数

-	函数对象到实例方法对象的转换每次获取实例该属性时都会发生
	-	有时可将属性赋值给本地变量、调用实现性能优化
	-	非用户定义函数、不可调用对象在被获取时不会发生转换
	-	实例属性的用户定义函数不会被转换为绑定方法，仅在函数
		是类的属性时才会发生

> - Py3中以上特性依赖于`___getattribute__`实现

####	属性

-	绑定方法对象支持只读获取底层函数对象任意属性
	-	但方法属性实际保存在下层函数对象中
	-	所以不能直接设置绑定方法的方法属性，必须在下层函数
		对象中显式设置，否则`raise AttributeError`

-	绑定方法有两个特殊只读属性
	-	`m.__self__`：操作该方法的类实例
	-	`m.__func__`：底层实现该方法的函数

> - `m(args,...)`完全等价于`m.__func__(m.__self__,args,...)`

####	特殊元属性

-	`__self__`：类对象实例
-	`__func__`：函数对象实例
-	`__doc__`：方法文档，等同于`__func__.__doc__`
-	`__name__`：方法名，等同于`__func__.__name__`
-	`__module__`：定义方法所在模块名

###	*Class Method Objects*

类方法：提供了**始终将类绑定为函数对象首个参数**的方式

-	对其他对象的封装，通常用于封装用户定义方法对象
-	会改变从类、类实例获取该对象的方式
-	用途
	-	实现自定义、多个构造器，如
		-	只调用`__new__()`、绕过`__init__`，创建未初始化
			的实例
		-	反序列化对象：从字节串反序列构造符合要求的对象

> - 通过`classmethod()`构造器创建
> - Py3中以上特性依赖于`___getattribute__`实现

###	*Static Method Objects*

静态方法：提供了**避免将函数对象转换为方法对象**的方式

-	对任意其他对象的封装，通常用于封装用户定义方法对象
-	从类、类实例获取静态方法对象时，实际返回的是封装的对象，
	不会被进一步转换
-	静态方法对象自身不是可调用的，但其封装的对象通常可调用

> - 通过内置`staticmethod()`构造器创建
> - Py3中以上特性依赖于`___getattribute__`实现

###	内置函数、方法

> - 内置函数：对C函数的外部封装
> - 内置方法：内置函数另一种形式，（类似实例方法）隐式传入
	当前实例作为C函数额外参数

-	包括以下两种类型
	-	`builtin_function_or_method`
	-	`wrapper_descriptor`
-	参数数量、类型由C函数决定
-	内置方法由支持其的类型描述

####	特殊元属性

-	`__self__`：`<module 'builtins' (built-in)>`
-	`__doc__`：函数/方法文档
-	`__name__`：函数/方法名
-	`__module__`：所在模块名

###	说明

-	函数是描述器，函数类`function`实现有`__get__`方法
-	`function.__get__`即将函数首个参数进行绑定，返回绑定方法

> - 参见*cs_python/py3ref/cls_special_methods*

##	*Generator Functions*

生成器函数：使用`yield`语句的函数、方法称为生成器函数

-	生成器函数调用时返回生成器迭代器对象，控制执行函数体

> - `yield`表达式参见*cs_python/py3ref/expressions*

###	*Generator-Iterator*

生成器迭代器：生成器函数执行得到的迭代器

-	实现有迭代器协议`__next__`的迭代器类实例不同于
	生成器迭代器，其仅实现迭代
	-	迭代执行过程即`__next__`函数调用，重入（状态维护）
		由类负责
	-	无不同迭代状态区别
	-	不会自动获得`.send`、`.throw`、`.close`等方法

-	或者说生成器迭代器是：利用yield表达式对迭代器的的简化
	实现，并预定义`.send`、`.throw`、`.close`方法

> - 迭代器协议参见*cs_python/py3ref/cls_special_methods*

####	执行过程

-	其某方法被调用时，生成器函数开始执行
-	执行到第一个yield表达式被挂起，保留所有局部状态
	-	局部变量当前绑定
	-	指令指针
	-	内部求值栈
	-	任何异常处理状态
-	返回`expression_list`
-	调用生成器某方法，生成函数继续执行
-	执行`return`、函数体执行完毕将`raise StopIteration`

> - 生成器表达式、yield表达式参见
	*cs_python/py3ref/expressions*

####	生成器迭代器状态

生成器在声明周期中有如下4中状态

-	`"GEN_CREATED"`：等待执行
-	`"GEN_RUNNING"`：正在执行，只有多线程中才能看到
-	`"GEN_SUSPENDED"`：在yield表达式处挂起状态
-	`"GEN_CLOSED"`：关闭状态

> - 可通过`inspect.getgeneratorstate()`方法查看

####	`__next__`

```python
def(pre) generator.__next__():
	pass
```

-	用途：开始生成器函数执行、或从上次执行yield表达式处恢复
	执行
	-	生成器函数通过`__next__`方法恢复执行时，yield表达式
		始终取值为`None`
	-	执行至下个yield表达式

-	返回值：生成器迭代器产生的下个值
	-	yield表达式中`expression_list`值
	-	若生成器没有产生下个值就退出，`raise StopIteration`

> - 此方法通常隐式通过`for`循环、`next()`函数调用

####	`send`

```python
def(pre) generator.send(value):
	pass
```

-	用途：恢复生成器函数执行并向其“发送”值`value`
	-	`value`参数作为yield表达式的结果
	-	执行至下个yield表达式

-	返回值：生成器迭代器产生的下个值
	-	yield表达式中`expression_list`值
	-	若生成器没有产生下个值就退出，`raise StopIteration`

-	说明
	-	若生成器中包含子迭代器，`send`传参数值将被传递给
		下层迭代器，若子迭代器没有合适接收方法、处理，将
		`raise AttributeError`、`raise TypeError`
	-	`.send`方法参数为`None`时实际不会有传递参数行为
		-	调用`.send()`启动生成器时，必须以`None`作为调用
			参数，因为此时没有可以接收值的yield表达式
		-	子迭代器中没有处理参数时，`.send(None)`也能正常
			工作

####	`throw`

```python
def(pre) generator.throw(type[, value[, traceback]]):
	pass
```

-	用途：在生成器**暂停位置处**引发`type`类型异常
	-	若异常被处理：则执行直至下个yield表达式
	-	若生成器函数没有捕获传入异常、或引发另一个异常：异常
		会被传播给调用者

-	返回值：返回生成器产生的下个值（若异常被处理）
	-	若生成器没有产生下个值就退出，`raise StopIteration`

-	说明
	-	若生成器中包含子迭代器，`throw`传入异常将被传递给
		子迭代器
	-	调用`throw`启动生成器，会在生成器函数开头引发错误

####	`close`

```python
def(pre) generator.close():
	pass
```

-	用途：在生成器函数暂停处`raise GeneratorExit`
	-	若之后生成器函数正常退出、关闭、引发`GeneratorExit`
		（生成器中未捕获该异常）：则关闭生成器并返回调用者
	-	若生成器继续产生值：则`raise RuntimeError`
	-	若生成器引发其他异常：则传播给调用者
	-	若生成器由于异常、或正常而退出：则无操作

###	Yield表达式--调用外部函数

> - 生成器函数执行yield表达式类似**调用外部函数**

-	当前函数栈保留当前状态、挂起
-	执行yield表达式“完毕”后，“返回”到调用处
-	yield表达式“返回值”取决于生成器恢复执行所调用方法
	-	`.__next__`：`for`、`next()`调用，返回`None`
	-	`.send()`：返回`.send()`参数
-	此时整体类似于
	-	主调函数调用生成器函数
	-	生成器函数调用yield表达式

###	生成器函数--协程

> - 生成器函数类似协程，也被称为*semi-coroutine*，是协程子集

-	相似点
	-	可`yield`多次，有多个入口点
	-	执行可以被挂起
	-	可在恢复执行时传递参数控制执行
-	不同点
	-	`yield`后控制权总是转移给生成器迭代器调用者，生成器
		函数不能控制`yield`后继续执行的位置
		（`yield`表达式仅仅传递值给父程序，并不是指定需要
		跳转的、平等的协程）

###	生成器函数--`try`

> - `try`结构中任何位置都允许yield表达式

-	若生成器在**销毁之前**没有恢复执行（引用计数为0、被垃圾
	回收），`try`结构中的yield表达式挂起可能导致`finally`
	子句执行失败

-	此时应由运行该异步生成器的事件循环、或任务调度器负责调用
	生成器-迭代器`close()`方法，从而允许任何挂起的`finally`
	子句得以执行

###	例

```python
def echo(value=None):
	print("execution start")
	try:
		while True:
			try:
				value = (yield value)
			except Exception as e:
				value = e
	finally:
		print("clean up when gen.close() is called")

if __name__ == "__main__":
	gen = echo(1)
	print(next(gen))		# 1
	print(next(gen))		# None
	print(gen.send(2))		# 2
	print(gen.throw(TypeError, "spam"))	# TypeError('spam',)
	gen.close()				# clean up...
```

##	*Coroutine Function*

协程函数：使用`async def`定义的函数、方法

-	调用时返回一个`coroutine`对象
-	可能包含`await`表达式、`async with`、`async for`语句
	-	即协程函数能在执行过程中挂起、恢复

> - 协程参见*cs_program/#todo*

###	*Coroutine Objects*

协程对象：属于*awaitable*对象

-	协程执行：调用`__await__`，迭代其返回值进行控制
	-	协程结束执行、返回时，迭代器`raise StopIteration`，
		异常的`value`属性将指向返回值
	-	等待协程超过一次将`raise RuntimeError`

-	若协程引发异常，其会被迭代器传播
	-	协程不应该直接引发未处理的`StopIteration`

> - 可等待对象参见*cs_python/py3ref/cls_special_methods*

####	`send`

```python
def(pre) coroutine.send(value):
	pass
```

-	用途：开始、恢复协程执行
	-	`value is None`：相当于等待`__await__()`返回迭代器
		结果下一项
	-	`value is not None`：此方法将委托给导致协程挂起的
		迭代器的`send()`方法

-	返回值：返回值、异常等同对`__await__()`返回值迭代结果

####	`throw`

```python
def(pre) coroutine.throw(type[, value[, traceback]]):
	pass
```

-	用途：在协程内引发指定异常
	-	此方法将委托给导致协程内挂起的迭代器的`throw()`方法
		，若其存在
	-	否则异常在挂起点被引发

-	返回值：返回值、异常等同对`__await__()`返回值迭代结果
	-	若异常未在协程内被捕获，则传回给主调者

####	`close`

```python
def(pre) coroutine.close():
	pass
```

-	用途：清理自身并退出
	-	若协程被挂起，此方法先被委托给导致该协程挂起的迭代器
		的`.close()`方法，若其存在
	-	然后在挂起点`raise GeneratorExit`，使得协程立即清理
		自身
	-	最后协程被标记为已结束执行，即使未被启动
	-	协程对象将被销毁时，会被自动调用

##	*Asynchronous Generator Functions*

异步生成器函数：包含`yield`语句、使用`async def`定义函数、
方法（即在协程中）

-	返回异步生成器迭代器对象，控制执行函数体

> - `yield from`表达式在异步生成器函数中使用将引发语法错误

###	*Asynchronous Generator-Iterator*

异步生成器迭代器

-	异步体现：`async def`定义协程函数中可以使用异步代码

-	异步生成器迭代器方法返回的可等待对象**们**执行同一函数栈
	-	可等待对象被`await`**运行时才会执行**异步函数体
	-	类似普通生成器迭代器，执行到yield表达式挂起
	-	则连续返回多个可等待对象可以乱序`await`

	> - 可以视为返回**待执行函数体**

> - 异步生成器迭代器执行过程、yield表达式、`try`结构、同异步
	迭代器协议关联， 均参见普通生成器迭代器

> - 异步迭代器协议参见*cs_python/py3ref/cls_special_methods*

> - 异步生成器函数使用`yield from`语句将引发语法错误

####	最终化处理#todo

处理最终化：事件循环应该定义终结器函数

-	其接收异步生成器迭代器、且可能调用`aclose()`方法并执行
	协程
-	终结器可以通过调用`sys.set_asyncgen_hooks()`注册
-	首次迭代时，异步生成器迭代器将保存已注册终结器以便最终化
	时调用

####	`__anext__`

```python
async def(pre) coroutine agen.__anext__():
	pass
```

-	用途：返回可等待对象，可等待对象运行时开始、恢复异步
	生成器函数执行
	-	异步生成器函数通过`__anext__`方法恢复执行时，返回的
		可等待对象中yield表达式始终取值为`None`

-	**可等待对象返回值**：返回异步生成器的下个值
	-	执行至下个yield表达式，返回`expression_list`值
	-	若异步生成器没有产生下个值就退出，则
		`raise StopAsyncIteration`，

> - 此方法通常由`async for`异步循环隐式调用

####	`asend`

```python
async def(pre) coroutine agen.asend(value):
	pass
```

-	用途：返回可等待对象，可等待对象运行时开始、恢复异步
	生成器函数执行，并向其“发送”值`value`
	-	`value`参数作为yield表达式的结果
	-	执行至下个yield表达式

-	返回值：异步生成器产生的下个值
	-	yield表达式中`expression_list`值
	-	若生成器没有产生下个值就退出，则
		`raise StopAsyncIteration`

-	说明
	-	调用`.asend()`启动生成器时，必须以`None`作为调用参数

####	`athrow`

```python
async def(pre) coroutine agen.athrow(type[, value[, traceback]]):
	pass
```

-	用途：返回可等待对象，可等待对象运行时将在异步生成器函数
	**暂停位置处**引发`type`类型异常
	-	若异常被处理：则执行直至下个yield表达式
	-	若异步生成器函数没有捕获传入异常、或引发另一个异常：
		可等待对象运行时异常会被传播给调用者

-	返回值：返回生成器产生的下个值（若异常被处理）
	-	若生成器没有产生下个值就退出，则
		`raise StopAsyncIteration`

-	说明
	-	调用`athrow`启动异步生成器，会在函数开头引发错误

####	`aclose`

```python
async def(pre) coroutine agen.aclose():
	pass
```

-	用途：返回可等待对象，可等待对象运行时在异步生成器函数
	暂停处`raise GeneratorExit`
	-	若异步生成器函数正常退出、关闭、引发`GeneratorExit`
		（生成器中未捕获该异常）：则运行可等待对象将
		`raise StopIteration`？？？
	-	后续调用异步生成器迭代器方法返回其他可等待对象：则
		运行可等待对象将`raise StopAsyncIteration`
	-	若异步生成器函数产生值：则运行可等待对象将
		`raise RuntimeError`
	-	若异步生成器迭代器已经异常、正常退出，则后续调用
		`aclose`方法将返回无行为可等待对象

