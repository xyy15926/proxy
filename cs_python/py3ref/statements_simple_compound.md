---
title: Simple Statements
categories:
  - Python
  - Py3Ref
tags:
  - Python
  - Py3Ref
  - Statements
date: 2019-06-11 15:56:31
updated: 2022-06-29 10:09:03
toc: true
mathjax: true
comments: true
description: Simple Statements
---

#	简单语句

-	简单语句：由单个逻辑行构成
	-	多条简单语句可以在同一物理行内、并以分号分隔

```bnf
simple_stmt ::=  expression_stmt
                 | assert_stmt
                 | assignment_stmt
                 | augmented_assignment_stmt
                 | annotated_assignment_stmt
                 | pass_stmt
                 | del_stmt
                 | return_stmt
                 | yield_stmt
                 | raise_stmt
                 | break_stmt
                 | continue_stmt
                 | import_stmt
                 | future_stmt
                 | global_stmt
                 | nonlocal_stmt
```

##	*Expression Statements*

```bnf
expresssion_stmt ::= starred_expression
```

-	表达式语句：用于计算、写入值（交互模式下），或调用过程（不返回有意义结果的函数）
	-	用途：表达式语句对指定表达式[列表]进行求值
	-	交互模式下
		-	若值不为 `None`：通过内置 `repr()` 函数转换为字符串，单独一行写入标准输出
		-	值为 `None`：不产生任何输出

##	*Assignment Statements*

```bnf
assignment_stmt ::=  (target_list "=")+ (starred_expression | yield_expression)
target_list     ::=  target ("," target)* [","]
target          ::=  identifier
                     | "(" [target_list] ")"
                     | "[" [target_list] "]"
                     | attributeref
                     | subscription
                     | slicing
                     | "*" target
```

-	赋值语句：将名称 *重绑定** 到特定值、修改属性或可变对象成员项
	-	用途：对指定表达式列表求值，将单一结果对象从左至右逐个赋值给目标列表
	-	赋值根据目标列表的格式递归定义
		-	目标为可变对象组成部分时（属性引用、抽取、切片），可变对象赋值有效性会被检查，赋值操作不可接受可能引发异常
	-	赋值顺序：将赋值看作是左、右端项重叠
		-	根据定义赋值语句内多个赋值是同时重叠，如：`a,b=b,a` 交换两变量值
		-	但赋值语句左端项包含集合类型时，重叠从左到右依次执行

```python
x = [0,1]
i = 0
i, x[i] = 1, 2					# `x` 被置为 `[0, 2]`，`i` 先被赋新值
```

###	赋值逻辑

-	赋值目标（左端项）为列表（可包含在圆括号、方括号内），按以下方式递归定义
	-	若目标列表为不带逗号、可以包含在圆括号内的单一目标，将右端项赋值给该目标
	-	否则
		-	若目标列表包含带 `*` 目标：右端项应为至少包含目标列表项数 -1 的可迭代对象
			-	加星目标前元素被右端项前段元素一一赋值
			-	加星目标后元素被右端项后段元素一一赋值
			-	加星目标被赋予剩余目标元素构成的列表（类似实参解包）
		-	否则
			-	右端项须为与目标列表相同项数的可迭代对象，其中元素将从左至右顺序被赋值给对应目标

-	赋值目标为单个目标，按以下方式递归定义
	-	目标为标识符（名称）
		-	名称未出现在当前代码块 `global`、`nonlocal` 语句中：名称被绑定到当前局部命名空间对象
		-	否则，名称被分别绑定到全局命名空间、或 `nonlocal` 确定的外层命名空间中对象
		-	若名称已经被绑定则被重新绑定，可能导致之前被绑定名称对象引用计数变为 0，对象进入释放过程并调用其析构器
	-	目标为属性引用
		-	引用中原型表达式被求值：应产生具有可赋值属性的对象，否则 `raise TypeError`
		-	向该对象指定属性将被赋值，若无法赋值将 `raise AttributeError`
	-	目标为抽取项（即 `Primary[Subsript]`）
		-	对 *Primary* 表达式求值：应产生可变序列对象（列表）、映射对象（字典）
		-	对 *Subscript* 表达式求值
			-	若 *Primary* 表达式为可变序列对象
				-	抽取表达式产生整数，包含负数则取模，结果只须为小于序列长度非负整数
				-	整数指定的索引号的项将被赋值
				-	若索引超出范围将 `raise IndexError`
			-	若 *Primary* 表达式为映射对象
				-	抽取表达式须产生与该映射键类型兼容的类型
				-	映射可以创建、更新抽取表达式指定键值对
		-	对用户自定义对象，将调用 `__setitem__` 方法并附带适当参数
	-	目标为切片
		-	对 *Primary* 表达式求值：应当产生可变序列对象
		-	上界、下界表达式若存在将被求值
			-	值应为应为正整数，包含负值则取模
			-	默认值分别为零、序列长度
		-	切片被赋值为右端项
			-	右端项应当是相同类型的序列对象
			-	若切片长度和右端项长度不同，将在目标序列允许情况下改变目标序列长度

###	*Augmented Assignment Statements*

```bnf
augmented_assignment_stmt ::=  augtarget augop (expression_list | yield_expression)
augtarget                 ::=  identifier | attributeref | subscription | slicing
augop                     ::=  "+=" | "-=" | "*=" | "@=" | "/=" | "//=" | "%=" | "**="
                               | ">>=" | "<<=" | "&=" | "^=" | "|="
```

-	增强赋值语句：在单个语句中将二元运算和赋值语句合为一体
	-	增强赋值语句不能类似普通赋值语句为可迭代对象拆包
	-	赋值增强语句对目标和表达式列表求值
		-	依据两操作数类型指定执行二元运算
		-	将结果赋给原始目标
	-	增强赋值语句可以被改写成类似、但非完全等价的普通赋值语句
		-	增强赋值语句中目标仅会被求值一次
		-	在可能情况下，运算是原地执行的，即直接修改原对象而
			不是创建新对象并赋值给原对象
		-	所以增强赋值**先对左端项求值**
	-	其他增强赋值语句和普通赋值语句不同点
		-	单条语句中对元组、多目标赋值赋值操作处理不同

###	*Annotated Assignment Statements*

```bnf
annotated_assignment_stmt ::=  augtarget ":" expression ["=" expression]
```

-	带标注的赋值语句：在单个语句中将变量、或属性标注同可选赋值赋值语句合并
	-	与普通赋值语句区别仅在于：仅适用仅包含单个目标的场合
	-	不同作用域中
		-	在类、模块作用域中
			-	若赋值目标为简单名称，标注被求值、存入类、模块的 `__annotations__` 属性中
			-	若赋值目标为表达式，标注被求值但不保存
		-	在函数作用域内，标注不会被求值、保存
	-	是否包含右端项
		-	若存在右端项，带标注赋值在对标注值求值前执行实际赋值
		-	否则，对赋值目标求值，但不执行 `__setitem__`、`__setattr__`（即计算至 *Primary* 为止）

> - Python3.8 中，带标注的赋值语句开始允许右端项为常规表达式，之前未带圆括号的元组表达式将导致语法错误

##	关键字语句

###	`assert`

```bnf
assert_stmt ::= "assert" expression ["," expression]
```

-	`assert` 语句：在程序中插入调试性断言的简便方式
	-	`assert` 语句等价于如下语句
	-	假定 `__debug__`、`AssertionError` 指向具有特定名称的内置变量，当前实现中
		-	对 `__debug__` 赋值是非法的，其值在解释器启动时确定
		-	默认内置变量 `__debug__=True`
		-	请求优化时`__debug__`置为`False`
			-	`-0` 命令行参数开启
			-	若编译时请求优化，代码生成器不会为 `assert` 语句生成代码
	-	无需再错误信息中包含失败表达式源码，其会被作为栈追踪一部分被显示

```python
if __debug__:						# 等价于`assert expression`
	if not expression:
		raise AssertionError
if __debug__:						# 等价于`assert expression1, expression2`
	if not expression1:
		raise AssertionError(expression2)
```

###	`pass`

```bnf
pass_stmt ::= "pass"
```

-	`pass` 语句：空操作，被执行时无事情发生
	-	适合语法上需要语句、但不需要执行代码时临时占位

###	`del`

```bnf
del_stmt ::= "del" target_list
```

-	`del` 语句：从局部、全局命名空间中移除名称的绑定，若名称未绑定将 `raise NameError`
	-	删除是递归定义的
		-	类似赋值的定义方式
		-	从左至右递归的删除目标列表中每个目标
	-	属性、抽取、切片的删除会被传递给相应 *Primary*
		-	删除切片基本等价于赋值为目标类型的空切片

###	`return`

```bnf
return_stmt ::= "return" [expression_list]
```

-	`return` 语句：离开当前函数调用，返回列表表达式值、`None`
	-	若提供表达式列表，其将被求值，缺省为`None`
	-	`return` 将控制流传出带有 `finally` 子句的 `try` 语句时，`finally` 子句会先被执行然后真正离开函数
	-	`return`语法上只会出现于函数定义代码块中，不会出现于类定义所嵌套代码中
		-	在生成器函数中，`return` 语句表示生成器已完成并`raise StopIteration`
			-	返回值作为参数构建 `StopIteration`，并作为 `StopIteration.value` 属性
		-	异步生成器函数中，`return` 语句表示异步生成器已完成并`raise StopAsyncIteration`
			-	非空`return`返回值在将导致语法错误

###	`yield`

```bnf
yield_stmt ::= yield_expression
```
-	`yield` 语句：语义上等于 `yield` 表达式
	-	可用于省略在使用等效yield表达式必须的圆括号
	-	`yeild` 表达式、语句仅在定义（异步）生成器函数时使用，且仅用于函数体内部，且函数体包含 `yield` 就使得该定义创建生成器函数而非普通函数

```python
yield <expr>
yield from <expr>			# 以上yield语句、以下yield表达式等价

(yield <expr>)
(yield from <expr>)
```

###	`raise`

```bnf
raise_stmt ::= "raise" [expression ["from" expression]]
```

-	`raise` 语句：引发异常
	-	若不带表达式：`raise` 将重新引发当前作用域最后一个激活异常
		-	若当前作用域内没有激活异常，将引发 `RuntimeError` 提示错误
	-	否则，计算表达式作为异常对象
		-	异常对象须为 `BaseException` 子类、实例
		-	若表达式结果为类，则通过不带参数的实例化该类作为异常实例
	-	异常被引发时会自动创建回溯对象，并被关联至可写 `__traceback__` 属性
		-	可以创建异常时同时使用 `.with_traceback()` 异常方法自定义回溯对象

```python
raise Exception("foo occured").with_traceback(tbobj)
```

-	`from` 子句用于异常串联
	-	`from` 后表达式求值需要为另一个异常类、实例
		-	其（或实例化后）将作为可写 `__cause__` 属性被关联到引发的第一个异常
		-	若引发异常未被处理，两个异常都将被打印
	-	新异常在已有异常被处理时被引发将隐式触发类似异常串联的机制
		-	异常处理即位于 `finally`、`except`、`with` 语句中
		-	此时，之前异常被关联到新异常的 `__context__` 属性
		-	可通过 `from None` 抑制

```python
try:
	print(1/0)
except:
	raise RuntimeError("Bad") from None			# 抑制 `print(1/0)` 导致的异常串联
```

###	`break`

```bnf
break_stmt ::= "break"
```

-	`break` 语句：终结最近外层循环、循环的可选 `else` 子句
	-	`break`在语法上只出现在 `for`、`while` 所嵌套代码
		-	不包括循环内函数定义、类定义所嵌套代码
		-	若 `for` 循环被 `break` 终结，循环控制目标保持当前值
	-	当 `break` 将控制流传出带有 `finally` 子句时，`finally` 子句会先被执行，然后真正离开循环

###	`continue`

```bnf
continue_stmt ::= "continue"
```

-	`continue` 语句：继续执行最近外层循环的下一轮次
	-	`continue` 在语法上只出现在 `for`、`while` 所嵌套代码
		-	不包括循环内函数定义、类定义、`finally` 子句所嵌套代码
	-	当 `continue` 将控制流传出带有 `finally` 子句时，`finally` 子句会先被执行，然后真正开始循环下一个轮次

###	`import`

```bnf
import_stmt     ::=  "import" module ["as" identifier] ("," module ["as" identifier])*
                     | "from" relative_module "import" identifier ["as" identifier]
                     ("," identifier ["as" identifier])*
                     | "from" relative_module "import" "(" identifier ["as" identifier]
                     ("," identifier ["as" identifier])* [","] ")"
                     | "from" module "import" "*"
module          ::=  (identifier ".")* identifier
relative_module ::=  "."* module | "."+
```

####	步骤、绑定

-	基本 `import` 语句（不包含 `from` 子句）执行步骤
	-	查找模块，若有必要加载并初始化模块
	-	为 `import` 所处作用域的局部命名空间定义名称
	-	包含多个子句（逗号分隔）时，以上两个步骤将分别对每个子句执行，如同子句被分成独立`import`语句

-	包含 `from` 子句的 `import` 语句执行步骤
	-	查找 `from` 子句中指定模块，若有必要则加载并初始化模块
	-	对 `import` 子句中指定的每个标识符
		-	检查被导入模块是否有该名称属性，若有则导入
		-	若没有，尝试导入具有该名称子模块，然后再次检查被导入（上级）模块是否具有该属性
			-	若未找到该属性，则 `raise ImportError`
			-	否则，将对该值引用存入局部命名空间，若有`as`子句则使用其指定名称，否则使用该属性名称

-	模块名称绑定
	-	若模块名称后带有 `as`，则在 `as` 之后名称将绑定到所导入模块
	-	若没有指定其他名称、且被导入模块为最高层级模块，则模块名称被绑定到局部命名空间作为对所导入模块的引用
	-	若被导入模块不是最高级模块，则
		-	包含该模块的最高层级包名将被绑定到局部命名空间作为的该最高层级包的引用
		-	所导入模块必须使用完整限定名称访问而不能直接访问

-	默认情况下，导入的父模块中命名空间中不包含子模块属性，即导入父模块**不能直接通过属性`.`引用子模块**
	-	有些包会在父模块中导入子模块，则初始化模块时父模块中即有子模块属性
	-	在当前模块手动导入子模块，子模块绑定至父模块命名空间中同名属性

####	通配符 `*`

-	标识符列表为通配符`*`形式
	-	模块中定义的全部公有名称都被绑定至 `import` 语句作用域对应局部命名空间
		-	通配符模式仅在模块层级允许使用，在类、函数中使用将 `raise SyntaxError`
	-	模块命名空间 `__all__` 属性指定模块定义的公有名称
		-	其中字符串项为模块中定义、导入的名称
		-	其中中所给出的名称被视为公有、应当存在
		-	应该包含所有公有API、避免意外导出不属于API部分项
		-	若 `__all__` 属性未定义，则公有名称集合将包括在模块命名空间中找到的、所有不以`_`开头名称

####	相对导入

```bnf
form ...sub_sub_pkg import mod1
```

-	相对导入：指定导入模块时，无需指定模块绝对名称
	-	需要导入的模块、包被包含在同一包中时，可在相同顶级包中进行相对导入，无需指明包名称
	-	在 `from` 子句中指定的模块、包中使用前缀点号指明需要上溯包层级数
		-	一个前缀点号：执行导入的模块在当前包
		-	两个前缀点号：上溯一个包层级
		-	三个前缀点号：上溯两个包层级，依此类推
	-	相对导入可以避免模块之间产生冲突，适合导入相关性强代码
		-	脚本模式（在命令行中执行 `.py` 文件）不支持相对导入
		-	要跨越多个文件层级导入，只需要使用多个 `.`

> - *PEP 328* 建议：相对导入层级不要超过两层

####	`future` 语句

```bnf
future_stmt ::=  "from" "__future__" "import" feature ["as" identifier]
                 ("," feature ["as" identifier])*
                 | "from" "__future__" "import" "(" feature ["as" identifier]
                 ("," feature ["as" identifier])* [","] ")"
feature     ::=  identifier
```

-	`future` 语句：指明某个特定模块使用在特定、未来某个 Python 发行版中成为标准特性的语法、语义
	-	用途
		-	允许模块在包含新特性发行版前使用该特性
		-	目的是使得在迁移至引入不兼容改变的 Python 未来版本更容易
	-	`future` 语句是针对编译器的指令 
		-	在编译时被识别并做特殊对待
			-	改变核心构造语义常通过生成不同代码实现
			-	新特性可能会引入不兼容语法，如：新关键字，编译器可能需要以不同方式解析模块
		-	编译器需要知道哪些特性名称已经定义
			-	包含未知特性的 `future` 语句将引发编译时错误
		-	直接运行时语义同其他 `import` 语句
			-	相应运行时语义取决于 `future` 语句启用的指定特性
	-	`future` 语句必须在靠近模块开头位置处出现，仅以下语句可出现在 `future` 语句前
		-	模块文档字符串
		-	注释
		-	空行
		-	其他 `future` 语句

-	Python3.9 中仅剩 `annotations` 特性需要使用 `future` 语句，其他历史上特性总是默认开启（为保持兼容性仍可冗余导入）
	-	`absolute_import`
	-	`division`
	-	`generators`
	-	`generartor_stop`
	-	`unicode_literals`
	-	`print_function`
	-	`nested_scope`

-	`future` 语句影响整个解释器环境
	-	在包含 `future` 语句的环境中，通过 `exec()`、`compile()` 调用代码会使用 `future` 语句关联的语法、语义
		-	此行为可以通过 `compile()` 可选参数加以控制
	-	`-i` 启动解释器并传入包含 `future` 语句的脚本，则交互式会话中将保持 `future` 语句有效

> - `import __future__ [as name]` 不是 `future` 语句，只是没有特殊语义、语法限制的普通 `import` 语句
###	`global`

```bnf
global_stmt ::= "global" identifier ("," identifier)*
```

-	`global` 语句：声明所列出标识符将被解读为全局变量
	-	`global` 语句是作用于整个当前代码块的声明
	-	局部作用域中给全局变量赋值必须用到 `global` 关键字
		-	仅仅是获取值无需 `global` 语句声明
		-	但自由变量也可以指向全局变量而不必声明为全局变量
	-	`global` 语句中列出的名称
		-	不能被定义为形参名
		-	不能作为 `for` 循环控制目标
		-	不能出现在类定义、函数定义、`import` 语句、变量标注中

> - CPython：暂时未强制要求上述限制，未来可能更改

-	`global` 是对 *Parser* 的指令，仅对与 `global` 语句同时被解析的代码起作用
	-	包含在作为 `exec()` 参数的字符串、代码对象中 `global` 语句不会影响 `exec()` 所在代码块
	-	反之，`exec()` 中代码也不会被调用其代码块影响
	-	`eval()`、`compile()` 等函数同

###	`nonlocal`

```bnf
nonlocal_stmt ::= "nonlocal" indentifier ("," identifier)*
```

-	`nonlocal` 语句：使得列出的名称指向之前最近的包含作用域中绑定的、除全局变量外的变量
	-	`nonlocal` 语句允许被封装代码重新绑定局部作用域以外、且非全局（模块）作用域当中变量
		-	即 `nonlocal` 语句中列出名称，必须指向之前存在于包含作用域中的绑定
	-	`nonlocal` 语句中列出名称不能与之前存在的局部作用域中绑定冲突

#	复合语句

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

-	复合语句：包含其他语句（语句组）的语句
	-	复合语句由一个、多个子句组成，子句包含句头、句体
		-	子句头
			-	都处于相同的缩进层级
			-	以作为唯一标识的关键字开始、冒号结束
		-	子句体
			-	在子句头冒号后、与其同处一行的一条或多条分号分隔的多条简单语句
			-	或者是在其之后缩进的一行、多行语句，此形式才能 **包含嵌套的复合语句**
	-	其会以某种方式影响、控制所包含的其他语句执行
	-	复合语句结构
		-	语句总以 `NEWLINE` 结束，之后可能跟随 `DEDENT`
		-	可选的后续子句总是以不能作为语句开头的关键字作为开头，不会产生歧义

##	关键字

###	`if` 语句

```bnf
if_stmt ::=  "if" expression ":" suite
             ("elif" expression ":" suite)*
             ["else" ":" suite]
```

-	`if` 语句：有条件的执行
	-	对表达式逐个求值直至找到真值，在子句体中选择唯一匹配者执行
	-	若所有表达式均为假值，则 `else` 子句体如果存在被执行

###	`while` 语句

```bnf
while_stmt ::=  "while" expression ":" suite
                ["else" ":" suite]
```

-	`while` 语句：在表达式保持为真的情况下重复执行
	-	重复检验表达式
		-	若为真，则执行第1个子句体
		-	若为假，则 `else` 子句体（若存在）就被执行并终止循环
	-	`break`、`continue` 语句
		-	第 1 个子句体中 `break` 语句执行将终止循环，且不执行 `else` 子句体
		-	第 1 个子句体中 `continue` 语句执行将跳过子句体中剩余部分，直接检验表达式

###	`for` 语句

```bnf
for_stmt ::= "for" target_list "in" expression_list ":" suite
	["else" : suite]
```

-	`for` 语句：对序列（字符串、元组、列表）或其他可迭代对象中元素进行迭代
	-	表达式列表被求值一次
		-	应该产生可迭代对象
		-	Python 将为其结果创建可迭代对象创建迭代器
	-	迭代器每项会按照标准赋值规则被依次赋值给目标列表
		-	为迭代器每项执行依次子句体
		-	所有项被耗尽 `raise StopIteration` 时，`else` 子句体（若存在）则会被执行
	-	目标列表中名称在循环结束后不会被删除
		-	但若序列为空，其不会被赋值
	-	`break`、`continue` 语句
		-	第 1 个子句体中 `break` 语句执行将终止循环，且不执行 `else` 子句体
		-	第 1 个子句体中 `continue` 语句执行将跳过子句体中剩余部分，直接检验表达式

-	注意事项
	-	序列在循环子句体中被修改可能导致问题
		-	序列的 `__iter__` 方法默认实现依赖内部计数器和序列长度的比较
		-	若在子句体中增、删元素会使得内部计数器“错误工作”
		-	可以对整个序列使用切片创建临时副本避免此问题

###	`try` 语句

```bnf
try_stmt  ::=  try1_stmt | try2_stmt
try1_stmt ::=  "try" ":" suite
               ("except" [expression ["as" identifier]] ":" suite)+
               ["else" ":" suite]
               ["finally" ":" suite]
try2_stmt ::=  "try" ":" suite
               "finally" ":" suite
```

-	`try` 语句：为一组语句指定异常处理器、清理代码

####	`except`子句

-	`except` 子句：指定一个、多个异常处理器
	-	`try` 子句中没有异常时，没有异常处理器执行
	-	否则，依次检查 `except` 子句直至找到和异常匹配的子句
		-	无表达式子句必须是最后一个，将匹配任何异常
		-	有表达式子句中表达式被求值，求值结果同异常兼容则匹配成功
			-	若在表达式求值引发异常
				-	则对原异常处理器搜索取消
				-	其被视为整个 `try` 语句引发异常，将在周边代码、主调栈中为新异常启动搜索
		-	若无法找到匹配的异常子句，则在周边代码、主调栈中继续搜索异常处理器

> - 兼容：是异常对象所属类、基类，或包含兼容异常对象元组

-	当找到匹配 `except` 子句时
	-	异常将被赋值给 `as` 子句后目标（若存在）
	-	对应子句体被执行（所有 `except` 子句都需要子句体）
	-	`as` 后目标在 `except` 子句结束后被清除
		-	避免因异常附加回溯信息而形成栈帧的循环引用，使得所有局部变量存活直至下次垃圾回收
		-	则异常必须赋值给其他名称才能在 `except` 子句后继续引用

```python
except E as N:
	foo

except E as N:			# `as` 目标被清除，即以上语句被转写为此
	try:
		foo
	finally:
		del N
```

> - `except` 子句执行前，异常信息存放在 `sys` 模块中，可通过 `sys.exc_info()` 访问

####	`else` 子句

-	`else`子句：以下情况均满足时将被执行（若存在）
	-	控制流离开 `try` 子句体没有引发异常
	-	没有执行 `return`、`continue`、`break` 语句

####	`finally` 子句

-	`finally` 子句：作为清理处理程序
	-	此子句体在任何情况下都被执行
		-	在 `try`、`except`、`else` 子句中引发的任何未处理异常将被临时保存
			-	执行完 `finally` 子句后被重新引发
			-	若 `finally` 子句中执行 `return`、`break` 语句，则临时保存异常被丢弃
			-	若 `finally` 子句引发新的异常，临时保存异常作为新异常上下文被串联
		-	`try` 子句中执行 `return`、`break`、`continue` 语句时，`finally` 子句在控制流离开 `try` 语句前被执行
			-	函数返回值由最后被执行的 `return` 语句决定，则 `finally` 子句中 `return`（若存在）将生效
		-	仅，`try` 语句中 `yield` 语句挂起后未恢复时，`finally` 子句将不执行
	-	执行期间程序不能获取任何异常信息

```bnf
def foo():
	try:
		return "try"
	finally:
		return "finally"		# 函数返回值由最后被执行的 `return` 语句决定
foo()							# 返回"finally"
```

###	`with`

```bnf
with_stmt ::=  "with" with_item ("," with_item)* ":" suite
with_item ::=  expression ["as" target]
```

-	`with` 语句：包装上下文管理器定义方法中代码块的执行
	-	`with` 句头中有多个项目，被视为多个 `with` 语句嵌套处理多个上下文管理器


```bnf
with A() as a, B() as b:
	suite
with A() as a:					# 上述多个项目等价于嵌套的多个上下文管理器
	wiht B() as b:
		suite
```

-	`with` 语句执行流程
	-	对表达式求值获得上下文管理器
	-	载入上下文管理器 `__exit__` 以便后续使用
	-	调用上下文管理器 `__enter__` 方法
	-	若包含 `as` 子句，`__enter__` 返回值将被赋值给其后目标
		-	`with` 语句保证若 `__enter__` 方法返回时未发生错误，`__exit__`总会被执行
		-	若在对目标列表赋值期间发生错误，视为在语句体内部发生错误
	-	执行 `with` 语句体
	-	调用上下文管理器 `__exit__` 方法
		-	若语句体退出由异常导致
			-	其类型、值、回溯信息将被作为参数传递给 `__exit__` 方法，否则提供三个 `None` 作为参数
			-	若 `__exit__` 返回值为假，该异常被重新引发，否则异常被抑制，继续执行 `with` 之后语句
		-	若语句体由于异常以外任何原因退出
			-	`__exit__` 返回值被忽略

```python
with EXPRESSION as TARGET:
	SUITE

manager = (EXPRESSION)					# 与上述 `with` 语句语义上等价
enter = type(manager).__enter__
exit = type(manager).__exit__			# 此即预先载入 `__exit__`
value = enter(manager)
hit_except = False

try:
    TARGET = value
    SUITE
except:
    hit_except = True
    if not exit(manager, *sys.exc_info()):
        raise
finally:
    if not hit_except:
        exit(manager, None, None, None)
```

##	*Function*

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

-	函数定义：对用户自定义函数的定义
	-	函数定义是可执行语句
		-	在当前局部命名空间中将函数名称绑定至函数对象（函数可执行代码包装器）
		-	函数对象包含对当前全局命名空间的引用以便调用时使用
	-	函数定义不执行函数体，仅函数被调用时才会被执行

###	*Decorators*

-	装饰器：函数定义可以被一个、多个装饰器表达式包装
	-	函数被定义时将在包含该函数定义作用域中对装饰器表达式求值，求值结果须为可调用对象
		-	其将以该函数对象作为唯一参数被调用
		-	返回值将被绑定至函数名称
	-	多个装饰器会以嵌套方式被应用
	-	装饰器包装类似将函数作为参数调用装饰器
		-	仅装饰器语句不涉及临时绑定函数名称

```bnf
@f1(arg)
@f2
def func():
	pass
def func():
	pass
func = f1(arg)(f2(func))		# 上述装饰器语句与函数调用大致等价，仅不会临时绑定名称
```

###	*Parameters*

-	形参类型
	-	*POSITIONAL_OR_KEYWORD*：之前没有 *VAR_POSITIONAL* 类型、`*` 的形参
		-	可以通过位置、关键字传值
	-	*KEYWORD_ONLY*：*VAR_POSITIONAL* 类型、或 `*` 的参数后形参
		-	只能通过关键字传值
	-	*POSITIONAL_ONLY*：`/` 前形参
		-	只能通过位置传值的参数，唯一不能使用关键字传参参数类型
		-	某些实现可能提供的函数包含没有名称的位置参数
	-	*VAR_POSITIONAL*：`*args` 星标形式参数
		-	只能通过位置传值
		-	多余位置实参填充于此，缺省为 `()`
	-	*VAR_KEYWORD*：`**kwargs` 双星标形式参数
		-	只能通过关键字传值
		-	多余关键字实参填充于此，缺省值为 `{}`

> - CPython：C 编写、`PyArg_ParseTuple()` 解析参数的函数常包含 *POSITIONAL_ONLY* 形参


####	*Default Parameters Values*

-	带默认参数值的形参：具有 `parameter = expression` 形式的形参
	-	具有默认值的形参，对应 `argument` 可以在调用中可被省略
	-	带默认值形参后、`*` 前形参必须具有默认值
	-	默认形参值将在执行函数定义时按从左至右的顺序被求值
		-	即，仅在函数定义时的表达式求值一次，之后每次调用时被使用
		-	则被作为默认值的列表、字典等可变对象将被所有未指定该参数调用共享，应该避免
			-	可以设置默认值为`None`，并在函数体中显式测试

####	*Annotations*

-	函数定义中可以添加标注
	-	标注格式
		-	形参标注：`param:expression`
		-	函数返回标注：`-> expression`
	-	标注不会改变函数语义
	-	标注可以是任何有效 Python 表达式
		-	默认在执行函数定义时被求值
		-	使用 `future` 表达式`from __future__ import annotations`，则标注在运行时被保存为字符串以启用延迟求值特性
	-	标注默认存储在函数对象 `__annotation__` 属性字典中
		-	可以通过对应参数名称、`"return"` 访问

##	*Class*

```bnf
classdef    ::=  [decorators] "class" classname [inheritance] ":" suite
inheritance ::=  "(" [argument_list] ")"
classname   ::=  identifier
```

-	类定义：对类的定义
	-	类定义为可执行语句
		-	继承列表 `inheritance` 通常给出基类列表、元类
		-	基类列表中每项都应当被求值为运行派生子类的类
		-	没有继承类列表的类默认继承自基类 `object`
	-	类定义语句执行过程
		-	类体将在新的执行帧中被执行
		-	使用新创建的局部命名空间和原有的全局命名空间
		-	类体执行完毕之后
			-	丢弃执行帧
			-	保留局部命名空间
		-	创建类对象
			-	给定继承列表作为基类
			-	保留的局部命名空间作为属性字典 `__dict__`
		-	类名称将在原有的全局命名空间中绑定至该类对象
	-	类可以类似函数一样被装饰
		-	装饰器表达式求值规则同函数装饰器
		-	结果被绑定至类名称

##	*Coroutine*

```bnf
async_funcdef ::=  [decorators] "async" "def" funcname "(" [parameter_list] ")"
                   ["->" expression] ":" suite
```

-	协程函数
	-	协程函数可以在多个位置上挂起（保存局部状态）、恢复执行
	-	协程函数体内部
		-	`await`、`async` 是保留关键字
		-	`await`表达式、`async for`、`async with` 只能在协程函数体内部使用
		-	使用 `yield from` 表达式将`raise SyntaxError`

###	`async for`语句

```bnf
async_for_stmt ::= "async" for_stmt
```

-	`async for`语句：允许方便的对异步迭代器进行迭代

```python
async for TARGET in ITER:
	...BLOCK1...
else:
	...BLOCK2...

iter = (ITER)					# 与上述 `async for` 在语义上等价于
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

-	`async with` 语句：允许方便使用异步上下文管理器

```python
async with EXPR as VAR:
	BLOCK

mgf = (EXPR)					# 与上述 `async with` 语义上等价于
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

