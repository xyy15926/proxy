#	*Simple Statements*

简单语句：由单个逻辑行构成

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

表达式语句：用于计算、写入值（交互模式下），或调用过程（不
返回有意义结果的函数）

```bnf
expresssion_stmt ::= starred_expression
```

-	用途：表达式语句对指定表达式[列表]进行求值
-	交互模式下
	-	若值不为`None`：通过内置`repr()`函数转换为字符串，
		单独一行写入标准输出
	-	值为`None`：不产生任何输出

##	*Assignment Statements*

赋值语句：将名称[重]绑定到特定值、修改属性或可变对象成员项

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

-	用途：对指定表达式列表求值，将单一结果对象从左至右逐个
	赋值给目标列表

-	赋值根据目标列表的格式递归定义，目标为可变对象组成部分时
	（属性引用、抽取、切片），可变对象赋值有效性会被检查，
	赋值操作不可接受可能引发异常

-	赋值顺序：将赋值看作是左、右端项重叠
	-	根据定义赋值语句内多个赋值是同时重叠，如：`a,b=b,a`
		交换两变量值
	-	但赋值语句左端项包含集合类型时，重叠从左到右依次执行

		```python
		x = [0,1]
		i = 0
		i, x[i] = 1, 2
			# `x`现在为`[0, 2]`
		```

		> - `LOAD_XXX`指令将右端项**从左到右依次压栈**
		> - `ROT_N`指令交换栈顶元素
		> - `STORE_XXX`指令将栈顶元素弹出**从左到右**依次给
			右端项赋值

		-	以上语句中，计算表达式`x[i]`前`i`已经被赋新值

		> - `dis.dis()`查看代码块指令

###	赋值目标为列表

赋值目标（左端项）为列表（可选包含在圆括号、方括号内）

-	若目标列表为不带逗号、可以包含在圆括号内的单一目标，将
	右端项赋值给该目标

-	否则：右端项须为**与目标列表相同项数**的可迭代对象，其中
	元素将从左至右顺序被赋值给对应目标

-	若目标列表包含带`*`元素：则类似实参解包，其须为可迭代
	对象，且右端项至少包含目标列表项数-1

	-	加星目标前元素被右端项前段元素一一赋值
	-	加星目标后元素被右端项后段元素一一赋值
	-	加星目标被赋予剩余目标元素**构成的列表**

###	赋值目标为单个目标

####	目标为标识符（名称）

-	名称未出现在当前代码块`global`、`nonlocal`语句中：名称
	被绑定到/赋值为当前局部命名空间对象

-	否则：名称被分别绑定到/赋值为全局命名空间、或`nonlocal`
	确定的外层命名空间中对象

> - 若名称已经被绑定则被重新绑定，可能导致之前被绑定名称
	对象引用计数变为0，对象进入释放过程并调用其析构器

####	目标为属性引用

-	引用中原型表达式被求值：应产生具有可赋值属性的对象，
	否则`raise TypeError`

-	该对象指定属性将被赋值，若无法赋值将
	`raise AttributeError`

####	目标为抽取项

-	引用中原型表达式被求值：应产生可变序列对象（列表）、
	映射对象（字典）

-	引用中抽取表达式被求值

	-	若原型表达式求值为可变序列对象
		-	抽取表达式产生整数，包含负数则取模，结果只须为
			小于序列长度非负整数
		-	整数指定的索引号的项将被赋值
		-	若索引超出范围将`raise IndexError`

	-	若原型表达式求值为映射对象
		-	抽取表达式须产生与该映射键类型兼容的类型
		-	映射可以创建、更新抽取表达式指定键值对

-	对用户自定义对象，将调用`__setitem__`方法并附带适当参数

####	目标为切片

-	引用中原型表达式被求值：应当产生可变序列对象
	-	右端项应当是相同类型的序列对象

-	上界、下界表达式若存在将被求值
	-	其应为整数，若为负数将对原型表达式序列长度求模，最终
		边界被裁剪至0、序列长度开区间中
	-	默认值分别为零、序列长度

-	切片被赋值为右端项
	-	若切片长度和右端项长度不同，将在目标序列允许情况下
		改变目标序列长度

####	*Augmented Assignment Statements*

增强赋值语句：在单个语句中将二元运算和赋值语句合为一体

```bnf
augmented_assignment_stmt ::=  augtarget augop (expression_list | yield_expression)
augtarget                 ::=  identifier | attributeref | subscription | slicing
augop                     ::=  "+=" | "-=" | "*=" | "@=" | "/=" | "//=" | "%=" | "**="
                               | ">>=" | "<<=" | "&=" | "^=" | "|="
```

> - 增强赋值语句不能类似普通赋值语句为可迭代对象拆包

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

####	*Annotated Assignment Statements*

带标注的赋值语句：在单个语句中将变量、或属性标注同可选赋值
赋值语句合并

```bnf
annotated_assignment_stmt ::=  augtarget ":" expression ["=" expression]
```

-	与普通赋值语句区别仅在于：仅有单个目标、且仅有单个右端项
	才被允许

-	在类、模块作用域中
	-	若赋值目标为简单名称，标注会被存入类、模块的
		`__annotations__`属性中
	-	若赋值目标为表达式，标注被求值但不会被保存

-	在函数作用域内，标注不会被求值、保存

-	若存在右端项，带标注赋值在对标注值求值前执行实际赋值；
	否则仅对赋值目标求值，不执行`__setitem__`、`__setattr__`

	> - 参见*cs_python/py3ref/#todo*

##	关键字语句

###	`assert`

assert语句：在程序中插入调试性断言的简便方式

```bnf
assert_stmt ::= "assert" expression ["," expression]
```
-	assert语句等价于

	```python
	if __debug__:
		if not expression:
			raise AssertionError
		# 等价于`assert expression`

	if __debug__:
		if not expression1:
			raise AssertionError(expression2)
		# 等价于`assert expression1, expression2`
	```

	-	无需再错误信息中包含失败表达式源码，其会被作为栈追踪
		一部分被显示

-	假定`__debug__`、`AssertionError`指向具有特定名称的内置
	变量，当前实现中

	-	对`__debug__`赋值是非法的，其值在解释器启动时确定
	-	默认内置变量`__debug__=True`
	-	请求优化时`__debug__`置为`False`
		-	`-0`命令行参数开启
		-	若编译时请求优化，代码生成器不会为assert语句生成
			代码

###	`pass`

pass语句：空操作，被执行时无事情发生

```bnf
pass_stmt ::= "pass"
```

-	适合语法上需要语句、但不需要执行代码时临时占位

###	`del`

del语句：从局部、全局命名空间中移除名称的绑定，若名称未绑定
将`raise NameError`


```bnf
del_stmt ::= "del" target_list
```

-	删除是递归定义的
	-	类似赋值的定义方式
	-	从左至右递归的删除目标列表中每个目标

-	属性、抽取、切片的删除会被传递给相应原型对象
	-	删除切片基本等价于赋值为目标类型的空切片？？？

###	`return`

return语句：离开当前函数调用，返回列表表达式值、`None`

-	在生成器函数中，return语句表示生成器已完成
	-	并`raise StopIteration`
	-	返回值作为参数构建`StopIteration`，并作为
		`StopIteration.value`属性

-	异步生成器函数中，return语句表示异步生成器已完成
	-	并`raise StopAsyncIteration`
	-	非空`return`返回值在将导致语法错误

```bnf
return_stmt ::= "return" [expression_list]
```

-	`return`语法上只会出现于函数定义代码块中，不会出现于
	类定义所嵌套代码中

-	若提供表达式列表，其将被求值；否则缺省为`None`

-	`return`将控制流传出带有`finally`子句的`try`语句时，
	`finally`子句会先被执行然后真正离开函数

###	`yield`

yield语句：语义上等于yield表达式

```bnf
yield_stmt ::= yield_expression
```

-	可用于省略在使用等效yield表达式必须的圆括号

	```python
	yield <expr>
	yield from <expr>
		# 以上yield语句、以下yield表达式等价

	(yield <expr>)
	(yield from <expr>)
	```

-	yeild表达式、语句仅在定义[异步]生成器函数时使用，且仅
	用于函数体内部，且函数体包含`yield`就使得该定义创建
	生成器函数而非普通函数

> - yield表达式参见*cs_python/py3ref/#todo*

###	`raise`

raise语句：引发异常

```bnf
raise_stmt ::= "raise" [expression ["from" expression]]
```

-	若不带表达式：`raise`将重新引发当前作用域最后一个激活
	异常
	-	若当前作用域内没有激活异常，将引发`RuntimeError`
		提示错误

-	否则计算表达式作为异常对象
	-	异常对象须为`BaseException`子类、实例
	-	若其为类，则通过不带参数的实例化该类作为异常实例

-	异常被引发时会自动创建回溯对象，并被关联至可写
	`__traceback__`属性

	-	可以创建异常时同时使用`.with_traceback()`异常方法
		自定义回溯对象

		```python
		raise Exception("foo occured").with_traceback(tbobj)
		```

####	异常串联

-	`from`子句用于异常串联：其后表达式求值需要为另一个异常

	-	其将作为可写`__cause__`属性被关联到引发的第一个异常
	-	若引发异常未被处理，两个异常都将被打印

-	若异常在try语句中`finally`子句、其他子句后、先中引发，
	类似机制发挥作用，之前异常被关联到新异常的`__context__`
	属性

> - 异常串联可以通过在`from`子句中指定`None`显式抑制

###	`break`

break语句：终结最近外层循环、**循环的可选`else`子句**

```bnf
break_stmt ::= "break"
```

-	`break`在语法上只出现在`for`、`while`所嵌套代码
	-	不包括循环内函数定义、类定义所嵌套代码

-	若`for`循环被`break`终结，循环控制目标保持当前值

-	当`break`将控制流传出带有`finally`子句的`try`语句时，
	`finally`子句会先被执行，然后真正离开循环

###	`continue`

continue语句：继续执行最近外层循环的下一伦茨

```bnf
continue_stmt ::= "continue"
```

-	`continue`在语法上只出现在`for`、`while`所嵌套代码
	-	不包括循环内函数定义、类定义、`finally`子句所嵌套
		代码

-	当`continue`将控制流传出带有`finally`子句的`try`语句时，
	`finally`子句会先被执行，然后真正开始循环下一个轮次

###	`import`

import语句

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

-	基本`import`子句执行步骤

	-	查找模块，若有必要加载并初始化模块
	-	为`import`所处作用域的局部命名空间定义名称

	> - 语句包含多个子句（逗号分隔）时，以上两个步骤将分别
		对每个子句执行，如同子句被分成独立`import`语句

-	默认情况下，导入的父模块中命名空间中不包含子模块属性，
	即导入父模块**不能直接通过属性`.`引用子模块**

	-	有些包会在父模块中导入子模块，则初始化模块时父模块
		中即有子模块属性
	-	在当前模块手动导入子模块，子模块绑定至父模块命名空间
		中同名属性

> - 导入机制参见*cs_python/py3ref/import_system*

####	绑定

-	若模块名称后带有`as`，则在`as`之后名称将直接绑定到所导入
	模块

-	若没有指定其他名称、且被导入模块为最高层级模块，则模块
	名称被绑定到局部命名空间作为对所导入模块的引用

-	若被导入模块不是最高级模块，则包含该模块的最高层级包名将
	被绑定到局部命名空间作为的该最高层级包的引用，所导入模块
	必须使用完整限定名称访问而不能直接访问

####	`from`子句

-	查找`from`子句中指定模块，若有必要则加载并初始化模块

-	对`import`子句中指定的每个标识符

	-	检查被导入模块是否有**该名称属性**
	-	若没有，尝试导入具有该名称子模块，然后再次检查
		**被导入（上级）模块**是否具有该属性
	-	若未找到该属性，则`raise ImportError`
	-	否则将对该值引用存入局部命名空间，若有`as`子句则
		使用其指定名称，否则使用该属性名称

#####	`*`通配符

标识符列表为通配符`*`形式：模块中定义的全部公有名称都被绑定
至`import`语句作用域对应局部命名空间

-	模块命名空间中`__all__`属性：字符串列表，指定模块
	**定义的公有名称**

	-	其中字符串项为模块中定义、导入的名称
	-	其中中所给出的名称被视为公有、应当存在
	-	应该包含所有公有API、避免意外导出不属于API部分项

-	若`__all__`属性未定义：则公有名称集合将包括在模块
	命名空间中找到的、所有不以`_`开头名称

> - 通配符模式仅在模块层级允许使用，在类、函数中使用将
	`raise SyntaxError`

#####	相对导入

相对导入：指定导入模块时，无需指定模块绝对名称

-	需要导入的模块、包被包含在同一包中时，可在相同顶级包中
	进行相对导入，无需指明包名称

-	在`from`子句中指定的模块、包中使用前缀点号指明需要上溯
	包层级数

	-	一个前缀点号：执行导入的模块在当前包
	-	两个前缀点号：上溯一个包层级
	-	三个前缀点号：上溯两个包层级，依此类推

		```bnf
		form ...sub_sub_pkg import mod1
		```

-	相对导入可以避免模块之间产生冲突，适合导入相关性强代码

	-	脚本模式（在命令行中执行`.py`文件）不支持相对导入
	-	要跨越多个文件层级导入，只需要使用多个`.`，但
		*PEP 328*建议，相对导入层级不要超过两层

####	*future*语句

future语句：指明莫格特定模块使用在特定、未来某个python发行版
中成为标准特性的语法、语义

```bnf
future_stmt ::=  "from" "__future__" "import" feature ["as" identifier]
                 ("," feature ["as" identifier])*
                 | "from" "__future__" "import" "(" feature ["as" identifier]
                 ("," feature ["as" identifier])* [","] ")"
feature     ::=  identifier
```

> - `import __future__ [as name]`：不是future语句，只是没有
	特殊语义、语法限制的普通import语句

-	用途
	-	允许模块在包含新特性发行版前使用该特性
	-	目的是使得在迁移至引入不兼容改变的python未来版本
		更容易

-	future语句是针对编译器的指令

	-	在编译时被识别并做特殊对待
		-	改变核心构造语义常通过生成不同代码实现
		-	新特性可能会引入不兼容语法，如：新关键字，编译器
			可能需要以不同方式解析模块

	-	编译器需要知道哪些特性名称已经定义
		-	包含未知特性的future语句将引发编译时错误

	-	直接运行时语义同其他import语句
		-	相应运行时语义取决于future语句启用的指定特性

	> - 在包含future语句的环境中，通过`exec()`、`compile()`
		调用代码**会使用future语句关联的语法、语义**，此行为
		可以通过`compile()`可选参数加以控制

-	future语句必须在靠近模块开头位置处出现，可以出现在future
	语句前的行

	-	模块文档字符串
	-	注释
	-	空行
	-	其他future语句

###	`global`

global语句：声明所列出标识符将被解读为全局变量

```bnf
global_stmt ::= "global" identifier ("," identifier)*
```

-	`global`语句是作用于整个当前代码块的声明

-	给全局变量赋值必须用到`global`关键字
	-	但自由变量也可以指向全局变量而不必声明为全局变量

-	global语句中列出的名称
	-	不能被定义为形参名
	-	不能作为`for`循环控制目标
	-	不能出现在类定义、函数定义、`import`语句、变量标注中

	> - CPython：暂时未强制要求上述限制，未来可能更改

-	`global`是对**解释器的指令**，仅对与global语句同时被解析
	的代码起作用

	-	包含在作为`exec()`参数的字符串、代码对象中global
		语句不会影响`exec()`所在代码块
	-	反之`exec()`中代码也不会被调用其代码块影响

	> - `eval()`、`compile()`等函数同

###	`nonlocal`

nonlocal语句：使得列出的名称指向**之前最近的包含作用域中**
绑定的、除全局变量外的变量

```bnf
nonlocal_stmt ::= "nonlocal" indentifier ("," identifier)*
```

-	nonlocal语句允许被封装代码重新绑定局部作用域以外、且非
	全局（模块）作用域当中变量

	-	即nonlocal语句中列出名称，必须指向**之前存在**于
		包含作用域中的绑定

	-	nonlocal语句中列出名称不能与之前存在的局部作用域中
		绑定冲突


