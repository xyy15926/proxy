#	*Compound Statements*

##	复合语句

复合语句：包含其他语句（语句组）的语句

-	复合语句由一个、多个子句组成，子句包含句头、句体
	-	子句头
		-	都处于相同的缩进层级
		-	以作为唯一标识的关键字开始、冒号结束
	-	子句体
		-	在子句头冒号后、与其同处一行的一条或多条分号分隔
			的多条简单语句
		-	或者是在其之后缩进的一行、多行语句，此形式才能
			**包含嵌套的复合语句**

-	其会以某种方式影响、控制所包含的其他语句执行

```bnf
compound_stmt ::=  if_stmt
                   | while_stmt
                   | for_stmt
                   | try_stmt
                   | with_stmt
                   | funcdef
                   | classdef
                   | async_with_stmt
                   | async_for_stmt
                   | async_funcdef
suite         ::=  stmt_list NEWLINE | NEWLINE INDENT statement+ DEDENT
statement     ::=  stmt_list NEWLINE | compound_stmt
stmt_list     ::=  simple_stmt (";" simple_stmt)* [";"]
```

> - 语句总以`NEWLINE`结束，之后可能跟随`DEDENT`
> - 可选的后续子句总是以不能作为语句开头的关键字作为开头，
	不会产生歧义

##	关键字

###	`if`

if语句：有条件的执行

```bnf
if_stmt ::=  "if" expression ":" suite
             ("elif" expression ":" suite)*
             ["else" ":" suite]
```

-	对表达式逐个求值直至找到真值，在子句体中选择唯一匹配者
	执行
-	若所有表达式均为假值，则`else`子句体如果存在被执行

###	`while`

while语句：在表达式保持为真的情况下重复执行

```bnf
while_stmt ::=  "while" expression ":" suite
                ["else" ":" suite]
```

-	重复检验表达式
	-	若为真，则执行第1个子句体
	-	若为假，则`else`**子句体存在**就被执行并终止循环

> - 第1个子句体中`break`语句执行将终止循环，且不执行`else`
	子句体
> - 第1个子句体中`continue`语句执行将跳过子句体中剩余部分，
	直接检验表达式

###	`for`

for语句：对序列（字符串、元组、列表）或其他可迭代对象中元素
进行迭代

```bnf
for_stmt ::= "for" target_list "in" expression_list ":" suite
	["else" : suite]
```

-	表达式列表被求值一次
	-	应该产生可迭代对象
	-	python将为其结果创建可迭代对象创建迭代器

-	迭代器每项会按照标准赋值规则被依次赋值给目标列表
	-	为迭代器每项执行依次子句体
	-	所有项被耗尽`raise StopIteration`时，`else`子句体
		存在则会被执行

-	目标列表中名称在循环结束后不会被删除
	-	但若序列为空，其不会被赋值

-	序列在循环子句体中被修改可能导致问题
	-	序列的`__iter__`方法默认实现依赖内部计数器和序列长度
		的比较
	-	若在子句体中增、删元素会使得内部计数器“错误工作”
	-	可以**对整个序列使用切片创建临时副本**避免此问题

> - 第1个子句体中`break`语句执行将终止循环，且不执行`else`
	子句体
> - 第1个子句体中`continue`语句执行将跳过子句体中剩余部分，
	转至下个迭代项执行

###	`try`

try语句：为一组语句指定异常处理器、清理代码

```bnf
try_stmt  ::=  try1_stmt | try2_stmt
try1_stmt ::=  "try" ":" suite
               ("except" [expression ["as" identifier]] ":" suite)+
               ["else" ":" suite]
               ["finally" ":" suite]
try2_stmt ::=  "try" ":" suite
               "finally" ":" suite
```

####	`except`子句

`except`子句：指定一个、多个异常处理器

-	`try`子句中没有异常时，没有异常处理器执行

-	否则，依次检查`except`子句直至找到和异常匹配的子句

	-	无表达式子句必须是最后一个，将匹配任何异常
	-	有表达式子句中表达式被求值，求值结果同异常兼容则匹配
		成功
		-	若在表达式求值引发异常，则对原异常处理器搜索取消
		-	其被视为整个`try`语句引发异常，将在周边代码、
			主调栈中为新异常启动搜索
	-	若无法找到匹配的异常子句，则在周边代码、主调栈中继续
		搜索异常处理器

	> - 兼容：是异常对象所属类、基类，或包含兼容异常对象元组


-	当找到匹配`except`子句时

	-	异常将被赋值给`as`子句后目标，若存在`as`子句
	-	对应子句体被执行（所有`except`子句都需要子句体）
	-	`as`后目标在`except`子句结束后被清除

		```python
		except E as N:
			foo
			# 被转写为
		except E as N:
			try:
				foo
			finally:
				del N
		```

		-	避免因异常附加回溯信息而形成栈帧的循环引用，使得
			所有局部变量存活直至下次垃圾回收
		-	则异常必须赋值给其他名称才能在`except`子句后继续
			引用

> - `except`子句体执行前，有关异常信息存放在`sys`模块中，
	参见*cs_python/py3std/os_sys.md*

####	`else`子句

`else`子句：在以下情况将被执行，若存在

-	控制流离开`try`子句体没有引发异常
-	没有执行`return`、`continue`、`break`语句

####	`finally`子句

`finally`子句：指定清理处理器，子句体在任何情况下都被执行

-	执行期间程序不能获取任何异常信息

	-	在`try`、`except`、`else`子句中引发的任何未处理异常
		将被临时保存，执行完`finally`子句后被重新引发

	-	但若`finally`子句中执行`return`、`break`语句，则临时
		保存异常被丢弃

	-	若`finally`子句引发新的异常，临时保存异常作为新异常
		上下文被串联

	> - 显式异常串联参见*cs_python/py3ref/simple_stmt*

-	`try`子句中执行`return`、`break`、`continue`语句时，
	`finally`子句在控制流离开try语句前被执行

	-	函数返回值由**最后被执行**的`return`语句决定，而
		`finally`子句总是最后被执行

		```bnf
		def foo():
			try:
				return "try"
			finally:
				return "finally"

		foo()
			# 返回"finally"
		```

###	`with`

`with`语句：包装上下文管理器定义方法中代码块的执行

```bnf
with_stmt ::=  "with" with_item ("," with_item)* ":" suite
with_item ::=  expression ["as" target]
```

-	`with`句头中有多个项目，被视为多个`with`语句嵌套处理多个
	上下文管理器

	```bnf
	with A() as a, B() as b:
		suite
		# 等价于
	with A() as a:
		wiht B() as b:
			suite
	```

####	执行流程

-	对表达式求值获得上下文管理器
-	载入上下文管理器`__exit__`以便后续使用
-	调用上下文管理器`__enter__`方法
-	若包含`as`子句，`__enter__`返回值将被赋值给其后目标
	-	`with`语句保证若`__enter__`方法返回时未发生错误，
		`__exit__`总会被执行
	-	若在对目标列表赋值期间发生错误，视为在语句体内部发生
		错误
-	执行`with`语句体
-	调用上下文关管理器`__exit__`方法
	-	若语句体退出由异常导致
		-	其类型、值、回溯信息将被作为参数传递给`__exit__`
			方法；否则提供三个`None`作为参数
		-	若`__exit__`返回值为假，该异常被重新引发；否则
			异常被抑制，继续执行`with`之后语句
	-	若语句体由于异常以外任何原因退出
		-	`__exit__`返回值被忽略

##	*Function*

函数定义：对用户自定义函数的定义

```bnf
funcdef                 ::=  [decorators] "def" funcname "(" [parameter_list] ")"
                             ["->" expression] ":" suite
decorators              ::=  decorator+
decorator               ::=  "@" dotted_name ["(" [argument_list [","]] ")"] NEWLINE
dotted_name             ::=  identifier ("." identifier)*
parameter_list          ::=  defparameter ("," defparameter)* ["," [parameter_list_starargs]]
                             | parameter_list_starargs
parameter_list_starargs ::=  "*" [parameter] ("," defparameter)* ["," ["**" parameter [","]]]
                             | "**" parameter [","]
parameter               ::=  identifier [":" expression]
defparameter            ::=  parameter ["=" expression]
funcname                ::=  identifier
```

-	函数定义是可执行语句
	-	在当前局部命名空间中将函数名称绑定至函数对象（函数
		可执行代码包装器）
	-	函数对象包含对当前全局命名空间的引用以便调用时使用

-	函数定义不执行函数体，仅函数被调用时才会被执行

###	*Decorators*

装饰器：函数定义可以被一个、多个装饰器表达式包装

-	函数被定义时将在包含该函数定义作用域中对装饰器表达式求值
	，求值结果须为可调用对象
-	其将以该函数对象作为唯一参数被调用；返回值将被绑定至函数
	名称
-	多个装饰器会以嵌套方式被应用

```bnf
@f1(arg)
@f2
def func():
	pass
	# 大致等价，仅以上不会临时绑定函数对象至名称
def func():
	pass
func = f1(arg)(f2(func))
```

###	*Parameter Types*

形参类型

-	*POSITIONAL_OR_KEYWORD*：之前没有*VAR_POSITIONAL*类型的
	参数
	-	可以通过**位置**、**关键字**传值

-	*KEYWORD_ONLY*：之前存在*VAR_POSITION*类型、或`*`的参数
	-	只能通过**关键字**传值

-	*VAR_POSITIONAL*：`*args`形式参数
	-	只能通过**位置**传值
	-	隐式默认值为`()`

-	*VAR_KEYWORD*：`**kwargs`形式参数
	-	只能通过关键字传值
	-	隐式默认值为`{}`

-	*POSITIONAL_ONLY*：只能通过位置传值的参数
	-	某些实现可能提供的函数包含没有名称的位置参数
	-	唯一不能使用关键字传参参数类型

	> - CPython：C编写、`PyArg_ParseTuple()`解析参数的函数

###	*Default Parameters Values*

默认参数值：具有`parameter = expression`形式的形参

-	具有默认值的形参，对应`argument`可以在调用中可被省略

-	默认形参值将在执行函数定义时按从左至右的顺序被求值
	-	即函数定义时的预计算值将在每次调用时被使用
	-	则被作为默认值的列表、字典等可变对象将被所有未指定该
		参数调用共享，应该避免
		-	可以设置默认值为`None`，并在函数体中显式测试

> - *POSITION_OR_KEYWORD*、有默认值形参必须位于无默认值者后
	，即若形参具有默认值，后续所有在`*`前形参必须具有默认值
> - *KEYWORD_ONLY*、有默认值形参可位于无默认值者前

###	*Annotations*

> - 形参标注：`param:expression`
> - 函数返回标注：`-> expression`

-	标注不会改变函数语义
-	标注可以是任何有效python表达式
	-	默认在执行函数定义时被求值
	-	使用future表达式`from __future__ import annotations`
		，则标注在运行时被保存为字符串以启用延迟求值特性
-	标注默认存储在函数对象`__annotation__`属性字典中
	-	可以通过对应参数名称、`"return"`访问

##	*Class*

类定义：对类的定义

```bnf
classdef    ::=  [decorators] "class" classname [inheritance] ":" suite
inheritance ::=  "(" [argument_list] ")"
classname   ::=  identifier
```

-	类定义为可执行语句
	-	继承列表`inheritance`通常给出基类列表、元类
	-	基类列表中每项都应当被求值为运行派生子类的类
	-	没有继承类列表的类默认继承自基类`object`

-	类定义语句执行过程
	-	类体将在新的执行帧中被执行
	-	使用新创建的局部命名空间和原有的全局命名空间
	-	类体执行完毕之后
		-	丢弃执行帧
		-	保留局部命名空间
	-	创建类对象
		-	给定继承列表作为基类
		-	保留的局部命名空间作为属性字典`__dict__`
	-	类名称将在原有的全局命名空间中绑定至该类对象

-	类可以类似函数一样被装饰
	-	装饰器表达式求值规则同函数装饰器
	-	结果被绑定至类名称

> - 类中属性、方法参见*#todo*
> - 类属性可以作为实例属性的默认值，但注意使用可变类型值可能
	导致未预期结果

##	*Coroutine*

协程函数

```bnf
async_funcdef ::=  [decorators] "async" "def" funcname "(" [parameter_list] ")"
                   ["->" expression] ":" suite
```

-	协程函数可以在多个位置上挂起（保存局部状态）、恢复执行
-	协程函数体内部
	-	`await`、`async`是保留关键字
	-	`await`表达式、`async for`、`async with`只能在协程
		函数体内部使用
	-	使用`yield from`表达式将`raise SyntaxError`

###	`async for`语句

```bnf
async_for_stmt ::= "async" for_stmt
```

-	`async for`语句允许方便的对异步迭代器进行迭代

	```python
	async for TARGET in ITER:
		...BLOCK1...
	else:
		...BLOCK2...
	```

	在语义上等价于

	```python
	iter = (ITER)
	iter = type(iter).__aiter__(iter)
	running = True
	while running:
		try:
			TARGET = await type(iter).__anext__(iter)
		except StopAsyncIteration:
			running = False
		else:
			BLOCK1
	else:
		BLOCK2
	```

###	`async with`语句

```bnf
async_with_stmt ::= "async" with_stmt
```

-	`async with`语句允许方便使用异步上下文管理器

	```python
	async with EXPR as VAR:
		BLOCK
	```

	语义上等价于

	```python
	mgf = (EXPR)
	aexit = type(mgr).__aexit__
	aenter = type(mgr).__aenter__(mgr)

	VAR = await aenter
	try:
		BLOCK
	except:
		if not await aexit(mgr, *sys.exc_info()):
			raise
	else:
		await aexit(mgr, None, None, None)
	```





