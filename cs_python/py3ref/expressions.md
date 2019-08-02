---
title: 表达式
tags:
  - Python
  - Py3Ref
categories:
  - Python
  - Py3Ref
date: 2019-06-06 00:04:02
updated: 2019-06-06 00:04:02
toc: true
mathjax: true
comments: true
description: 表达式
---

##	*Atoms*

原子：表达式最基本元素

-	最简单原子
	-	标识符
	-	字面值

-	以圆括号、方括号、花括号包括的形式在语法上也被归为原子

```bnf
atom      ::=  identifier | literal | enclosure
enclosure ::=  parenth_form | list_display | dict_display | set_display | generator_expression | yield_atom
```

###	*Indentifiers/Names*

名称：作为原子出现的标识符

-	名称被绑定到对象时：对原子求值将返回相应对象
-	名称未绑定时：对原子求值将`raise NameError`

####	*Private Name Mangling*

> - 类的私有名称：文本形式出现在类定义中以两个、更多下划线
	开头且不以两个、更多下划线结尾的标识符

私有名称转换：在为私有名称生成代码前，其被转换为更长形式

-	转换方式：在名称前插入类名、下划线
	-	若转换后名称太长（超过255字符），**某些实现中**可能
		发生截断

-	转换独立于标识符使用的句法

-	若类名仅由下划线组成，则不会进行转换

###	*Literals*

字面值：

```bnf
literal ::=  stringliteral | bytesliteral | integer | floatnumber | imagnumber
```

-	对字面值求值将返回改值对应类型的对象
	-	对浮点数、复数，值可能为近似值

-	所有字面值都对应不可变数据类型
	-	所以对象标识的重要性不如其实际值

-	多次对具有相同值的字面值求值，可能得到相同对象、或具有
	相同值的不同对象

	-	元组是不可变对象，适用字面值规则：两次出现的空元组
		产生对象可能相同、也可能不同
#todo

###	*Parenthesized Forms*

带括号形式：包含在`()`的**可选表达式列表**

```bnf
parenth_form ::= "(" [starred_expression] ")"
```

-	带圆括号表达式列表将返回表达式所产生的任何东西
	-	内容为空的圆括号返回空元组对象
	-	列表包含至少一个逗号，产生元组
	-	否则，产生表达式列表对应的单一表达式

-	元组不是由圆括号构建，实际是`,`逗号操作符起作用
	-	空元组是例外，此时圆括号必须，因为表达式中不带圆括号
		的“空”会导致歧义

###	*List/Set/Dict/Generator-Tuple*

> - *display*：显式列出容器内容
> - *comprehension*：推导式，通过循环、筛选指令计算

```bnf
comprehension ::=  expression comp_for
comp_for      ::=  ["async"] "for" target_list "in" or_test [comp_iter]
comp_iter     ::=  comp_for | comp_if
comp_if       ::=  "if" expression_nocond [comp_iter]
```

-	推导式结构：单独表达式后加至少一个`for`子句以及零个、或
	多个`for`或`if`子句
	-	`for`、`if`子句视为代码块，按**从左到右顺序嵌套**
		（类似`for`循环嵌套）
	-	每次**到达最内层代码块时**对表达式求值以产生元素

-	除最左边`for`子句中可迭代表达式，推导式在另一个隐式嵌套
	作用域内执行
	-	确保赋给目标列表的名称不会“泄露”到外层作用域
	-	最左边`for`子句中可迭代表达式直接在外层作用域中被
		求值，然后作为参数传递给隐式嵌套作用域
	-	后续`for`子句、最左侧`for`子句中任何筛选条件不能在
		外层作用域中被求值，因为其可能依赖于从最左侧可迭代
		对象中获得的值

-	为确保推导式总能得到类型正确的容器，隐式嵌套作用域内禁止
	使用`yield`、`yield from`表达式，因为其会对外层作用域
	造成附加影响

-	若推导式包含`async for`子句、`await`表达式，则为异步
	推导式

####	*List Displays*

列表显式：用`[]`方括号括起的、可能为空的表达式系列

```bnf
list_display ::= "[" [starred_list | comprehesion] "]"
```

-	列表显式会产生新的列表对象，内容通过表达式、推导式指定
-	提供逗号分隔的表达式时：元素从左至右求值，按此顺序放入
	列表对象
-	提供推导式时：根据推导式产生结果元素进行构建

####	*Set Displays*

集合显式：用`{}`花括号标明，与字典区别在于没有冒号分隔键值

```bnf
set_display ::=  "{" (starred_list | comprehension) "}"
```

-	集合显式产生可变集合对象，内容通过表达式、推导式指定
-	提供逗号分隔的表达式时：元素从左至右求值，按此顺序放入
	列表对象
-	提供推导式时：根据推导式产生结果元素进行构建

> - 空集合不能使用`{}`构建，此构建的是空字典


####	*Dict Displays*

字典显式：用`{}`花括号括起来的、可能为空的键值对

```bnf
dict_display       ::=  "{" [key_datum_list | dict_comprehension] "}"
key_datum_list     ::=  key_datum ("," key_datum)* [","]
key_datum          ::=  expression ":" expression | "**" or_expr
dict_comprehension ::=  expression ":" expression comp_for
```

> - `**`：映射拆包，操作数必须是映射

-	字典显式产生新的字典对象

-	提供`,`分隔键值对序列
	-	从左至右被求值以定义字典条目
	-	可多次指定相同键，最终值由最后给出键值对决定

-	提供字典推导式
	-	以冒号分隔的两个表达式，后者带上标准`for`、`if`子句
	-	作为结果键值对按产生顺序被加入新字典

> - 键类型需要为hashable

###	*Generator Expression*

生成器表达式：用圆括号括起来的紧凑形式生成器（迭代器）标注

```bnf
generator_expression ::=  "(" expression comp_for ")"
```

-	生成器表达式会产生新的生成器（迭代器）对象
	-	句法同推导式，但使用圆括号括起
	-	圆括号在只附带一个参数（省略`expression`）的调用中
		可以被省略

-	生成器表达式中使用的变量在生成器对象调用`__next__`方法
	时以惰性方式被求值，同普通生成器
	-	最左侧`for`子句内可迭代对象会被立即求值，则其造成的
		错误会在生成器表达式**被定义时被检测到**

###	*Yield Expression*

yield表达式：将控制权交还给调度程序

```bnf
yield_atom       ::=  "(" yield_expression ")"
yield_expression ::=  "yield" [expression_list | "from" expression]
```

> - `yield`：返回其后表达式求值
> - yield表达式是赋值语句右侧唯一表达式时，括号可以省略

-	在定义生成器函数、异步生成器函数时才会用到，也只能在函数
	定义内部使用yield表达式，将函数变为（异步）生成器函数

-	yield表达式会对外层作用域造成附带影响，不允许作为实现
	推导式、生成器表达式隐式作用域的一部分

> - 生成器、异步生成器参见*cs_python/py3ref/dm_gfuncs*

####	`yield from`

`yield from`：其后表达式视为子迭代器，将控制流委托给其

-	类似**管道**，迭代器参数、异常都被传递给子迭代器
	-	**子迭代器依次迭代结果**被传递给生成器方法调用者
	-	`.send`传递值、`.throw`生成异常被传递给子迭代器

-	`.send`传入值、`.throw`传入异常如果有适当方法将被传递给
	下层迭代器，否则
	-	`send`将`raise AttributeError`、`raise TypeError`
	-	`throw`将立即引发传入异常

-	子迭代器完成后引发的`StopIteration`实例的`value`属性将
	作为yield表达式值
	-	可以在引发`StopIteration`时被显式设置#todo
	-	在子迭代器是生成器时通过从子生成器返回值自动设置

-	用途
	-	展开嵌套序列

##	*Primaries*

原型：代表编程语言中最紧密绑定的操作（优先级最高）

```bnf
primary ::= atom | attributeref | subscription | slicing | call
```

###	*Attributeref*

属性引用：后面带有句点加名称的原型

```bnf
attributeref ::= primary "." identifier
```

-	要求值为支持属性引用类型的对象（多数对象支持）
-	对象会被要求产生以指定标识符为名称的属性
	-	产生过程可以通过重载`__getattr__()`方法自定义

###	*Subscriptions*

抽取：在序列（字符串、元组、列表）、映射（字典）对象中选择
一项

```bnf
subscription ::= primary "[" expression_list "]"
```

-	要求值必须为支持抽取操作的对象
	-	可以定义`__getitem__()`方法支持抽取操作

-	映射：表达式列表求值须为键值
	-	抽取操作选择映射中键对应值
	-	表达式列表为元组，除非其中只有一项

-	序列：表达式列表求值须为整数、或切片
	-	正式句法规则没有要求实现对负标号值处理，但内置序列
		`__getitem__()`方法结合序列长度解析负标号
	-	重载`__getitem__`的子类需要显式添加对负标号、切片
		支持

###	*Slicings*

切片：在序列对象（字符串、元组、列表）中选择某个范围内的项

```bnf
slicing      ::=  primary "[" slice_list "]"
slice_list   ::=  slice_item ("," slice_item)* [","]
slice_item   ::=  expression | proper_slice
proper_slice ::=  [lower_bound] ":" [upper_bound] [ ":" [stride] ]
lower_bound  ::=  expression
upper_bound  ::=  expression
stride       ::=  expression
```

-	可以用作表达式赋值、`del`语句的目标

-	形似表达式列表的东西同样形似切片列表，所以任何抽取操作
	都可以被解析为切片
	-	通过定义将此情况解析为抽取优先于切片以消除歧义

-	原型使用`__getitem__`、根据切片列表构造的键进行索引
	-	切片列表包含逗号：键将为包含切片项转换的元组
	-	否则：键为单个切片项的转换
	-	切片项若为表达式：切片的转换即为切片对象

###	*Calling*

调用：附带可能为空的一系列参数来执行可调用对象

```bnf
call                 ::=  primary "(" [argument_list [","] | comprehension] ")"
argument_list        ::=  positional_arguments ["," starred_and_keywords]
                            ["," keywords_arguments]
                          | starred_and_keywords ["," keywords_arguments]
                          | keywords_arguments
positional_arguments ::=  ["*"] expression ("," ["*"] expression)*
starred_and_keywords ::=  ("*" expression | keyword_item)
                          ("," "*" expression | "," keyword_item)*
keywords_arguments   ::=  (keyword_item | "**" expression)
                          ("," keyword_item | "," "**" expression)*
keyword_item         ::=  identifier "=" expression
```

-	要求值为可调用对象
	-	用户定义函数
	-	内置函数
	-	内置对象方法
	-	类对象
	-	类实例方法
	-	任何具有`__call__()`方法的对象

-	调用流程
	-	参数表达式在尝试调用前被求值
	-	所有参数表达式被转换为参数列表
	-	代码块将形参绑定到对应参数表达式值

-	除非引发异常，调用总有返回值
	-	返回值可能为`None`
	-	返回值计算方式取决于可调用类型
		-	用户定义函数、实例方法、类实例：函数返回值
		-	内置函数：依赖于编译器
		-	内置对象方法：类新实例

> - 在位置参数、关键字参数后加上括号不影响语义

####	关键字实参转位置实参

若存在关键字实参，会通过以下操作被转换为位置参数

-	为正式参数创建**未填充空位的列表**
-	若有N个位置参数：将其放入前N个空位
-	对每个关键字参数
	-	使用标识符确定对应的空位：若标识符与第k个正式
		参数名相同，使用第k个空位
	-	若空位已被填充：则`raise TypeError`
	-	否则将参数值放入空位进行填充
-	所有参数处理完毕后，未填充空位使用默认值填充
-	若仍有未填充空位，则`raise TypeError`；否则**填充完毕**
	列表被作为调用的参数列表

####	多余实参

-	若有关键字参数没有与之对应的正式参数名称，将
	`raise TypeError`，除非有形参使用`**indentifier`句法
	-	`identifier`将被初始化新的有序映射接收任何额外关键字
		参数
	-	若没有多余关键字实参，则为相同类型空映射

-	若位置实参数目多余位置形参数目，将`raise TypeError`，
	除非有形参使用`*identifier`句法
	-	`identifier`将初始化为元组接受任何额外位置参数
	-	没有多余位置实参，则为空元组

####	实参解包

-	若实参中出现`*expression`句法

	-	`expression`求值须为`iterable`
	-	来自该可迭代对象的元素被当作额外位置实参
	-	`*expression`可以放在关键字实参后而没有语法错误
		-	`expression`会优先被迭代，元素用于填充参数列表
		-	可能和关键字参数冲突，导致关键字参数对应空位被
			填充

		> - 一般位置实参必须位于关键字实参前，否则有语法错误

-	若实参中出现`**expresion`句法

	-	`expression`求值须为`mapping`
	-	其内容被当作额外关键字参数
		-	若关键字已存在，将`raise TypeError`

##	运算符

-	运算符优先级：从低到高

	|运算符|描述|
	|-----|-----|
	|`lambda`|lambda表达式|
	|`if--else`|条件表达式|
	|`or`|布尔逻辑或|
	|`and`|布尔逻辑与|
	|`not`|布尔逻辑非|
	|`in`、`not in`、`is`、`is not`、`<`、`<=`、`>`、`=>`、`!=`、`==`|比较运算，包括成员检测、标识号检测|
	|`|`|按位或|
	|`^`|按位异或|
	|`&`|按位与|
	|`<<`、`>>`|移位|
	|`+`、`-`|加、减|
	|`*`、`@`、`/`、`//`、`%`|乘、矩阵乘、除、整除、取余（字符串格式化）|
	|`+x`、`-x`、`~x`|正、负、按位取反|
	|`**`|幂次|
	|`await`|await表达式|
	|`x[index]`、`x[start:end]`、`x(arguments...)`|抽取、切片、调用、属性调用|
	|`(expression...)`、`[expressions...]`、`{key:value}`、`{expressions...}`|绑定或元组、列表、字典、集合显示|

-	求值顺序：从左至右对表达式求值
	-	但赋值操作时，右侧先于左侧求值

-	算术类型转换
	-	若任意参数为复数，另一参数转换为复数
	-	否则，若任意参数为浮点数，另一参数为浮点数
	-	否则，二者均为整数，不需要转换

###	`await`

`await`：挂起coroutine执行以等待awaitable对象

```bnf
await_expr ::= "await" primary
```

-	只能在协程函数中使用


###	幂运算符

幂运算符

```bnf
power ::= (await_expr | primary) ["**" u_expr]
```

-	优先级高于左侧一元运算符、低于右侧一元运算符

	```python
	-1 ** 2 == -1
	0 ** 0 == 1
		# 编程语言得普遍做法
	```

-	语义同两个参数调用内置`power`函数
	-	左参数进行右参数所指定的幂次乘方运算
	-	数值参数会转换为相同类型，返回转换后类型
		-	`int`类型做负数幂次：参数转换为`float`
		-	`0`进行负数幂次：`raise ZeroDivisionError`
		-	负数进行分数次幂次：返回`complex`类型

###	一元算术、位运算

一元算术、位运算

```bnf
u_expr ::= power | "-" u_expr | "+" u_expr | "~" u_expr
```

-	一元算数、位运算具有相同优先级
-	若参数类型不正确将`raise TypeError`
	-	`+`：产生数值参数相同的值
	-	`-`：产生数值参数的负值
	-	`~`：只作用于整数，对整数参数按位取反，返回`-(x+1)`
		（即负值使用补码存储）

###	二元算术运算符

二元算术运算符

```bnf
m_expr ::=  u_expr | m_expr "*" u_expr | m_expr "@" m_expr |
            m_expr "//" u_expr | m_expr "/" u_expr |
            m_expr "%" u_expr
a_expr ::=  m_expr | a_expr "+" m_expr | a_expr "-" m_expr
```

-	二元算术运算符遵循传统优先级，除幂运算符外只有两个优先
	级别
	-	乘法型
	-	加法型

-	python支持混合算术，二元运算符可以用于不同类型操作数
	-	精度较低者会被扩展为另一个操作数类型

####	算符说明

-	`@`：目标是用于矩阵乘法，没有内置类型实现此运算符

-	`%`：模，输出第1个参数除以第2个参数的余数

	-	参数可以是浮点数
	-	结果正负总是与第2个操作数一致、或为0
	-	结果绝对值一定小于第2个操作数绝对值（数学上必然真，
		但对浮点数而言由于舍入误差存在，数值上未必真）

-	`//`：整除，结果就是`floor`函数处理算术除法`/`的结果

	-	整除、模语义同内置函数`divmod(x,y) == (x//y, x%y)`
	-	若`x`接近`y`的整数倍，由于舍入误差的存在，`x//y`可能
		大于`(x-x%y)//y`，此时python返回后一个结果，保证
		`divmod(x,y)[0]*y + x % y`尽量接近`x`

-	某些运算符也作用于特定非数字类型

	-	`*`：两个参数分别为整数、序列，执行**序列重复**
	-	`%`：被字符串对象重载，用于执行旧式字符串格式化/插值
	-	`+`：两个参数为相同类型序列，执行序列拼接操作

###	移位运算

移位运算

```bnf
shift_expr ::= a_expr | shift_expr ("<<" | ">>") a_expr
```

-	优先级低于算术运算
-	运算符接受整数参数
	-	将第一个参数左移、右移第二个参数指定的bit数
	-	右移：`x >> n == x // power(2, n)`
	-	左移：`x << n == x * power(2, n)`

###	二元位运算

二元位运算

```bnf
and_expr ::=  shift_expr | and_expr "&" shift_expr
xor_expr ::=  and_expr | xor_expr "^" and_expr
or_expr  ::=  xor_expr | or_expr "|" xor_expr
```

-	三种位运算符具有不同的优先级

-	两个参数须为整数
	-	`&`：对两个参数进行按位*AND*运算
	-	`^`：对两个参数进行按位*XOR*运算
	-	`|`：对两个参数进行按位*OR*运算

###	比较运算

比较运算

```bnf
comparison    ::=  or_expr (comp_operator or_expr)*
comp_operator ::=  "<" | ">" | "==" | ">=" | "<=" | "!="
                   | "is" ["not"] | ["not"] "in"
```

-	所有比较运算优先级相同（与C不同）
	-	低于任何算术、移位、位运算

-	比较运算可以任意串联`a  op1 b op2 c ... y opN z`等价于
	`a op1 b and b op2 c and ... and y opN z`

	-	只是后者中每个表达式最多只被求值一次

	-	例：`a < b >= c`类似表达式会被按照传统比较法则解读
		-	等价`a < b and b >= c`
		-	仍具有短路求值特性，`a < b == false`时，`c`不会
			被求值

####	值比较

-	`>`、`<`、`==`、`!=`、`<=`、`>=`比较两个对象值
	-	不要求两个对象为相同类型

	> - 比较运算符实现了**特定对象值概念**，可以认为是通过
		实现对象比较间接定义对象值

-	所有类型继承于`object`，从其继承了默认比较行为

	-	`=`、`!=`：一致性比较，基于对象标识`id`
		-	具有相同标识的实例一致性比较结果相等
		-	此默认行为动机：希望对象都应该是自反射，即
			`x is y`就意味着`x == y`

	-	`<`、`>`、`<=`、`>=`：没有默认值提供
		-	尝试比较`raise TypeError`
		-	此默认行为动机：缺少一致性比较类似固定值

#####	数字类型

数字类型：`int`、`float`、`complex`以及标准库类型
`fractions.Fraction`、`decimal.Decimal`
	
-	可进行类型内部、跨类型比较

	-	类型相关限制内按数学（算法）规则正确进行比较，且不会
		有精度损失
	-	复数不支持次序比较

-	非数字值`float('NaN')`、`decimal.Decimal('NaN')`

	-	同任何其他数字值比较均返回`False`
	-	**不等于自身**，但是是同一个对象（标识相同）

	> - 具体实现参见*cs_python/py3ref/#todo*

#####	二进制码序列

二进制码序列：`bytes`、`bytearray`

-	可以进行类型内部、跨类型比较
-	使用元素数字值按字典序进行比较

#####	字符串

字符串：`str`

-	使用字符的Unicode码位数字值、按字典序比较
-	字符串、二进制码序列不能直接比较

#####	序列

序列：`tuple`、`list`、`range`

-	只能进行类型内部比较
	-	跨类型：一致性比较结果为否、次序比较将
		`raise TypeError`
	-	`range`不支持次序比较

-	序列元素通过相应元素进行字典序比较

	-	序列相等：相同类型、相同长度，每对相应元素必须 相等
	-	对支持次序比较序列：排序同第一个不相等元素排序，若
		对应元素不同，较短序列排序较小

-	强制规定元素自反射性：序列元素`x`，`x==x`总为真

	-	即序列元素比较比较时：须首先比较元素标识，仅会对不同
		元素执行`==`严格比较运算

	-	若序列元素为自反射元素，结果与严格比较相同；若序列
		元素为非自反射元素，结果与严格比较不同

		```python
		nan = float("NaN")
		(nan is nan) == True
		(nan == nan) == False
		([nan] == [nan]) == True
		```

#####	映射

映射：`dict`

-	映射相等：当且进行具有相同键值对
	-	键、值一致性比较强制规定自反射性

#####	集合

集合：`set`、`frozenset`

-	可进行类型内部、跨类型比较

-	比较运算符定义为子集、超集检测

	-	这类关系没有定义完全排序，如：`{1,2}`、`{2,3}`集合
		不相等，也没有大小比较关系
	-	所以，集合不应作为依赖完全排序的函数参数，如：
		`min`、`max`、`sorted`，将产生未定义结果

-	集合强制规定其元素自反射性

#####	自定比较行为

> - 其他内置类型没有实现比较方法，继承`object`默认比较行为
> - 可以通过实现**富比较方法**自定义类型的比较行为，最好
	遵守一些一致性规则（不强制）

-	自反射：相同对象比较应该相等
	-	`x is y`有`x == y`

-	对称性
	-	`x == y`有`y == x`
	-	`x != y`有`y != x`
	-	`x < y`有`y > x`
	-	`x <= y`有`y >= x`

-	可传递
	-	`x > y and y > z`有`x > z`
	-	`x < y and y <= z`有`x < z`

-	反向比较应该导致布尔取反
	-	`x == y`有`not x != y`
	-	`x < y`有`not x >= y`（对完全排序）
	-	`x > y`有`not x <= y`（对完全排序）

-	相等对象应该具有相同hash值，或标记为不可hash

> - 具体实现参见*cs_python/py3ref/#todo*

####	成员检测

`in`、`not in`：成员检测，后者为前者取反

-	对`list`、`tuple`、`set`、`frozenset`、`dict`、
	`collections.deque`等**内置**容器类型
	
	-	`x in y`同`any(x is e or x == e for e in y)`
	-	映射检测是否有给定键

-	对字符串、字节串

	-	当且进当`x`为`y`其子串时`x in y`返回`True`，空字符串
		总被视为其他字符串子串
	-	`x in y`等价于`y.find(x) != -1`

-	自定义类型可以自定义成员检测

	-	`__contains__`方法返回值即为`x in y`返回值
	-	未定义`__contains__`方法但定义`__iter__`，若迭代`y`
		得到值`z == x`，则`x in y == True`，出现异常等同于
		`in`引发异常

	> - 具体实现参见*cs_python/py3ref/#todo*

####	标识号比较

`is`、`is not`：对象标识号检测，后者为前者取反

-	当且仅当`x`、`y`为同一对象`x is y == True`

###	布尔运算

布尔运算

```bnf
or_test  ::=  and_test | or_test "or" and_test
and_test ::=  not_test | and_test "and" not_test
not_test ::=  comparison | "not" not_test
```

-	执行布尔运算、表达式用于流程控制语句时，以下值被解析为
	假值，其余值被解析为真值

	-	`False`
	-	`None`
	-	所有数值类型的数值0
	-	空字符串
	-	空容器

-	`and`、`or`**返回最终求值参数**而不是`False`、`True`

	-	`x and y`：首先对`x`求值
		-	对`x`求值，若为假直接返回`x`**求值**
		-	否则对`y`**求值并返回**

	-	`x or y`：首先对`x`求值
		-	对`x`求值，若为真直接返回`x`**求值**
		-	否则对`y`**求值并返回结果值**

-	`not`必须创建新值，无论参数为和类型均范围布尔值`True`、
	`False`

-	可以通过自定义`__bool__`方法定制逻辑值

	> - 具体实现参见*cs_python/py3ref/#todo*

###	条件表达式

条件表达式：三元运算符

```bnf
conditional_expression ::=  or_test ["if" or_test "else" expression]
expression             ::=  conditional_expression | lambda_expr
expression_nocond      ::=  or_test | lambda_expr_nocond
```

-	在所有python运算中具有最低优先级
-	`x if C else y`
	-	首先对条件`C`求值
	-	若`C`为真，`x`被求值并返回
	-	否则将对`y`求值并返回

###	`lambda`表达式

`lambda`表达式：创建匿名函数

```bnf
lambda_expr        ::=  "lambda" [parameter_list] ":" expression
lambda_expr_nocond ::=  "lambda" [parameter_list] ":" expression_nocond
```

-	`lambda parameters: expression`返回函数对象，同

	```python
	def <lambda>(parameters):
		return expression
	```

> - `lambda`表达式只是简单函数定义的简单写法

###	表达式列表

表达式列表：除作为列表、集合显示的一部分，包含至少一个逗号
的列表表达式将生成元组

```bnf
expression_list    ::=  expression ("," expression)* [","]
starred_list       ::=  starred_item ("," starred_item)* [","]
starred_expression ::=  expression | (starred_item ",")* [starred_item]
starred_item       ::=  expression | "*" or_expr
```

> - 末尾逗号仅在创建单独元组时需要，在其他情况下可选

-	元组长度就是列表中表达式的数量
	-	表达式将从左至右被求值

-	`*`表示可迭代拆包：操作数必须为*iterable*（同实参调用）
	-	可迭代对象将被拆解为迭代项序列，并被包含于新建的元组
		、列表、集合中


