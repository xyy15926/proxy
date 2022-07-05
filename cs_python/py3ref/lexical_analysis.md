---
title: Lexical Analysis
categories:
  - Python
  - Py3Ref
tags:
  - Python
  - Py3Ref
  - Grammer
  - Lexical
date: 2019-08-01 01:38:59
updated: 2022-07-05 10:07:03
toc: true
mathjax: true
comments: true
description: Lexical Analysis
---

##	综述

###	Python实现

-	Python 只是语言，具体解释器实现有很多种
	-	*CPython*：C 语言实现，最原始版本
		-	通常就被称为 Python，其他实现区分时才强调为 CPython
		-	新语言特性通常较早出现
	-	*Jython*：Java实现
		-	将Python代码编译为 Java 字节码，在 JVM 上运行
		-	可以左线Java应用的脚本语言、创建需要 Java 类库支持的应用
	-	PyPy：RPython（Python 语言子集）实现
		-	*JIT* 编译器，执行效率高于 CPython
		-	非栈式支持
		-	允许方便修改解释器，鼓励对语言本身进行实验

-	语言的具体实现可能发生改变、其他实现可能使用不同方式
	-	在语言的参考文档中加入过多细节实现很危险
	-	Python 是 CPython 解释器先于语言规范出现，且语言的设计哲学即实用优先

> - CPython 是 C 实现的解释器，`cython` 是将 Python 代码翻译为 C 插件的项目/包

###	*Notation*说明

```lex
name ::= lc_letter(lc_letter | "_")*
lc_letter ::= "a"..."z"
```

-	标注：词法、句法解析的描述使用修改过的 *BNF* 语法标注
	-	`::=`：声明规则，左侧为规则名称
	-	`|`：分隔可选项
	-	`*`：前一项的零次、多次重复
	-	`+`：前一项的一次、多次重复
	-	`[]`：括起内容可选，即出现零次、一次
	-	`()`：分组
	-	`""`：固定字符串包含在引号内
	-	` `：空格仅用于分隔 token
	-	`...`：三个点分割的本义字符表示在指定区间范围内的任意单个字符
	-	`<>`：对所定义符号的非常描述，在必要时用于说明“控制字符”意图

-	标注说明
	-	每条规则通常为一行，多个可选项规则可用`|`为界分为多行
	-	词法定义：作用域输入源中的单独字符
	-	句法定义：作用于词法分析生成的*token stream*

-	Python 将读取的程序文件转换为 Unicode 码点
	-	源文件的文本编码可由编码声明指定，默认 *UTF-8*
	-	词法分析器：将文件拆分为 token
	-	解释器：以词法分析器生成的 token 流作为输入

##	Python 顶层组件

###	完整的Python程序

-	完整的 Python 程序会在最小初始化环境中被执行
	-	所有内置、标准模块均可用，但均处于未初始化状态
		-	`sys`、`builtins`、`__main__` 除外（已经初始化）
		-	`__main__` 模块为完整程序的执行提供局部、全局命名空间
	-	Python 解释器可从以下三种源获取完整程序
		-	`-c` 命令行选项传递字符串
		-	作为首个命令行参数的文件
		-	标准输入

```bnf
file_input ::= (NEWLINE|statement)*
interactive_input ::= [stmt_list] NEWLINE | compound_stmt NEWLINE
eval_input ::= expression_list NEWLINE*
```

-	不同输入具有适用不同语法
	-	文件输入：从非交互式文件读取的输入具有相同形式，适合
		-	解析完整的 Python 程序（从文件、字符串）
		-	解析模块
		-	解析传递给 `exec()` 函数的字符串
	-	交互式输入：若文件、标准输入是 *tty* 设备，解释器进入交互模式
		-	初始环境同完整程序
		-	每次读取执行一条语句，语句会在 `__main__` 命名空间中被执行
			-	交互模式中（最高层级）复合语句后必须带有空行，帮助解释器确定输入的结束
	-	表达式输入
		-	`eval()` 被用于表达式输入
		-	表达式开头空白被忽略

##	行结构

###	逻辑行、物理行、行拼接

-	逻辑行：逻辑行的结束以 `NEWLINE` token 表示
	-	语句不能跨越逻辑行边界，除非其语法允许包含 `NEWLINE`（如复合语句包含多个子语句）
	-	逻辑行可由一个、多个物理行按照显式、隐式行拼接规则构成

-	物理行：以行终止序列结束的字符序列
	-	源文件、字符串中可以使用任何标准平台上行终止序列
		-	*Unix*：`LF` 换行符（C 标准转义 `\n`）
		-	*Win*：`CR LF` 回车加换行
		-	*Macintosh*：`CR` 回车（C 标准转义 `\r`）
	-	输入结束 `CTRL-D` 也可被作为最后物理行的隐含终止标志（常用于交互式）
	-	嵌入 Python 源码字符串应使用标准 C 传统换行符 `\n`

-	显式行拼接：两个、多个物理行使用 `\` 拼接为一个逻辑行
	-	物理行以不在字符串、注释内的反斜杠结尾时，将与下行拼接构成一个单独逻辑行
		-	反斜杠、其后换行符将被删除
	-	说明
		-	以反斜杠结束的行不能带有注释
		-	反斜杠不能用于
			-	拼接注释
			-	拼接字符串外 token
		-	不允许原文字符串以外反斜杠存在于物理行其他位置

-	隐式行拼接
	-	圆括号、方括号、花括号内表达式允许被分为多个物理行，无需使用反斜杠
	-	说明
		-	拼接行可以带有注释
		-	后续行缩进不影响程序结构、允许为空白行
		-	拼接行之间不会有 `NEWLINE` token
	-	三引号 `"""`、`'''` 字符串支持隐式拼接行

-	三引号`"""`/`'''`字符串允许被分为多个物理行
	-	拼接行中不允许带有注释

###	空白行、注释、编码生命

-	空白行：只包含空格符、制表符、进纸符、注释的逻辑行
	-	空白行会被忽略，不生成 `NEWLINE` token
	-	交互式输入语句时，对空白行处理可能因为 *读取-求值-打印* 循环的具体实现而存在差异
		-	标准交互模式解释器中：完全空白逻辑行将会结束一条多行复合语句

-	注释：以不包含在字符串内的 `#` 开头，在物理行末尾结束
	-	注释标志逻辑行的结束，除非存在隐含行拼接规则
	-	注释在语法分析中被忽略，不属于 token

-	编码声明：位于 Python 脚本第一、第二行，匹配正则表达式 `coding[=:]\s*([-\w.]+)` 的注释将被作为编码声明处理
	-	编码声明须独占一行，若在第二行，则第一行也必须为注释
	-	表达式第一组指定了源码文件编码
		-	编码声明指定编码名称必须是 Python 所认可的编码
		-	词法分析将使用此编码：语义字符串、注释、标识符
		-	默认编码为 *UTF-8*
		-	若，文件首字节为 *UTF-8* 字节顺序标志 `b\xef\xbb\xbf`，文件编码也声明为 *UTF-8*
	-	示例
		-	`# -*- coding: <encoding-name> -*-`：*GNU Emacs* 风格
		-	`# vim:fileencoding=<encoding-name>`：*VIM* 风格


###	缩进、空白符

-	缩进：**逻辑行开头** 的空白（空格符、制表符）被用于计算该行的缩进等级，决定语句段落组织结构
	-	首个非空白字符之前的空格总数确定该行的缩进层次
		-	缩进不能使用反斜杠进行拼接，首个反斜杠之前空格将确定缩进层次
	-	制表符：被替换为 1-8 个空格，使得缩进的空格总数为 8 倍数
		-	源文件若混合使用制表符、空格符缩进，并使得确定缩进层次需要依赖于制表符对应空格数量设置，将引发 `TabError`
	-	进纸符
		-	在行首时：在缩进层级计算中被忽略
		-	行首空格内其他位置：效果未定义，可能导致空格计数重置为 0
	-	兼容性说明
		-	由于非 Unix 平台上文本编辑器本身特性，不应在源文件中混合使用制表符、空格符
		-	不同平台可能会显式限制最大缩进层级

-	`INDENT`、`DEDENT`：连续行的缩进层级以堆栈形式生成 `INDENT`、`DEDENT` token
	-	读取文件第一行前，向堆栈中压入零值，不再弹出
	-	被压入栈的层级数值从底至顶持续增加
	-	每个逻辑行开头的行缩进层级将和栈顶进行比较
		-	相同：不做处理
		-	新行层级较高：压入栈中，生成 `INDENT` token
		-	新行层级较低：应为栈中层级数值之一
			-	栈中高于该层级所有数值被弹出
			-	每弹出一级数值生成一个 `DEDENT` token
	-	文件末尾，栈中剩余每个大于 0 数值生成一个 `DEDENT` token


##	*Token*

-	*Token* 型符
	-	空白字符不属于 token
		-	除逻辑行开头、字符串内，空格符、制表符、进纸符等空白符均可分隔 token
	-	存在二义性，将从左至右尽可能长读取合法字符串组成 token

-	其他一些 token
	-	字面值：内置类型常量值的表示方法

###	*Indentifier*

-	标识符（名称）：Python 标识符句法基于 *Unicode* 标准附件 *UAX-31*
	-	*ASCII* 字符集内有效标识符字符（同 Python2）
		-	大、小写字母
		-	下划线`_`
		-	数字`0-9`
	-	此外，Python3.0 中引入 *ASCII* 字符集以外的额外字符
		-	其分类使用包含于 `unicodedata` 模块中 *Unicode* 字符数据库版本
	-	标识符没有长度限制、大小写敏感

```lex
identifier   ::=  xid_start xid_continue*
id_start     ::=  <all characters in general categories Lu, Ll, Lt, Lm, Lo, Nl, the underscore, and characters with the Other_ID_Start property>
id_continue  ::=  <all characters in id_start, plus characters in the categories Mn, Mc, Nd, Pc and others with the Other_ID_Continue property>
xid_start    ::=  <all characters in id_start whose NFKC normalization is in "id_start xid_continue*">
xid_continue ::=  <all characters in id_continue whose NFKC normalization is in "id_continue*">
```

> - `Lu`：大写字母
> - `Ll`：小写字母
> - `Lt`：词首大写字母
> - `Lm`：修饰字母
> - `Lo`：其他字母
> - `Nl`：字母数字
> - `Mn`：非空白标识
> - `Mc`：含空白标识
> - `Nd`：十进制数字
> - `Pc`：连接标点
> - `Other_ID_Start`：由 `PropList.txt` 定义的显式字符列表，用来支持向后兼容
> - `Other_ID_Continue`：同上

####	*Keywords* 

-	关键字：以下标识符作为语言的保留字、关键字，不能用作普通标识符

```lex
False      await      else       import     pass
None       break      except     in         raise
True       class      finally    is         return
and        continue   for        lambda     try
as         def        from       nonlocal   while
assert     del        global     not        with
async      elif       if         or         yield
```

####	保留标识符类

-	某些以下划线字符开头、结尾的标识符类，具有特殊含义
	-	`_*`：不会被 `from module import *` 导入
		-	特殊标识符 `_`：交互式解释器中用于存放最近一次求值结果
		-	不处于交互模式时无特殊含义、无预定义
	-	`__*__`：系统定义名称
		-	由解释器极其实现（包括标准库）定义
		-	任何不遵循文档指定方式使用 `__*__` 行为可能导致无警告出错
	-	`__*`：类私有名称
		-	在类定义中使用
		-	会以混合形式重写避免基类、派生类私有属性之间出现名称冲突

###	字符串、字节串字面值

```lex
stringliteral   ::=  [stringprefix](shortstring | longstring)
stringprefix    ::=  "r" | "u" | "R" | "U" | "f" | "F"
                     | "fr" | "Fr" | "fR" | "FR" | "rf" | "rF" | "Rf" | "RF"
shortstring     ::=  "'" shortstringitem* "'" | '"' shortstringitem* '"'
longstring      ::=  "'''" longstringitem* "'''" | '"""' longstringitem* '"""'
shortstringitem ::=  shortstringchar | stringescapeseq
longstringitem  ::=  longstringchar | stringescapeseq
shortstringchar ::=  <any source character except "\" or newline or the quote>
longstringchar  ::=  <any source character except "\">
stringescapeseq ::=  "\" <any source character>
```

```lex
bytesliteral   ::=  bytesprefix(shortbytes | longbytes)
bytesprefix    ::=  "b" | "B" | "br" | "Br" | "bR" | "BR" | "rb" | "rB" | "Rb" | "RB"
shortbytes     ::=  "'" shortbytesitem* "'" | '"' shortbytesitem* '"'
longbytes      ::=  "'''" longbytesitem* "'''" | '"""' longbytesitem* '"""'
shortbytesitem ::=  shortbyteschar | bytesescapeseq
longbytesitem  ::=  longbyteschar | bytesescapeseq
shortbyteschar ::=  <any ASCII character except "\" or newline or the quote>
longbyteschar  ::=  <any ASCII character except "\">
bytesescapeseq ::=  "\" <any ASCII character>
```

> - `stringprefix`、`bytesprefix`与字面值剩余部分之间不允许由空白符
> - 源字符集由编码声明定义
> - 字节串字面值只允许 *ASCII* 字符（但允许存储不大于256）

-	两种字面值都可以使用成对（连续三个）单引号、双引号标示首尾
	-	单引号`''`：允许包含双引号`""`
	-	双引号`""`：允许包含单引号`''`
	-	三重引号`'''`、`"""`
		-	原样保留：未经转义的换行、（非三联）引号、空白符

-	字符（字节）串前缀
	-	`b`、`B`前缀：字节串字面值
		-	创建 `bytes` 类型而非 `str` 类型实例
		-	只能包含 *ASCII* 字符
		-	字节对应数值大于 128 必须以转义形式表示
	-	`r`、`R`：原始字符串、字节串
		-	其中反斜杠 `\` 被当作其本身字面字符处理
		-	转换序列不在有效
		-	原始字面值不能以单个 `\` 结束，会转义之后引号字符
	-	`f`、`F`：格式化字符串字面值

####	拼接

-	字符串字面值拼接
	-	多个相邻字符串、字符串字面值（空白符分隔），含义等同于全部拼接为一体
		-	所用引号可以彼此不同（三引号风格也可用）
		-	每部分字符串可以分别加注释
		-	可以包括格式化字符串字面值
	-	此特性是在句法层面定义，在编译时实现
		-	在运行时拼接字符串表达式必须使用`+`运算符

####	转义

-	转义规则：反斜杠 `\` 用于对特殊含义字符进行转义
	-	字符串、字节串字面值中转义序列基本类似标准 C 转义规则解释
	-	以下转义序列仅在字符串字面值中可用
		-	`\N{NAME}`：Unicode 中名称为 `NAME` 的字符
		-	`\uxxxx`：必须接受 4 个 16 进制数码
		-	`\Uxxxxxxxx`：必须接受 8 个 16 进制数码
	-	对无法识别的转义序列
		-	Python3.6 之前将原样保留在字符串中
		-	Python3.6 开始，将 `raise DeprecationWarning`，未来可能会 `raise SyntaxError`


####	格式化字符串字面值

```lex
f_string          ::=  (literal_char | "{{" | "}}" | replacement_field)*
replacement_field ::=  "{" f_expression ["!" conversion] [":" format_spec] "}"
f_expression      ::=  (conditional_expression | "*" or_expr)
                         ("," conditional_expression | "," "*" or_expr)* [","]
                       | yield_expression
conversion        ::=  "s" | "r" | "a"
format_spec       ::=  (literal_char | NULL | replacement_field)*
literal_char      ::=  <any code point except "{", "}" or NULL>
```

-	格式化字符串字面值：带有 `f/F` 前缀的字符串字面	格式表达式被当作正常的包含在圆括号中 Python 表达式处理，在运行时从左至右被求值
		-	不允许空表达式
		-	`lambda` 空表达式必须显式加上圆括号
		-	可以包含换行：如三引号字符串
		-	不能包含注释
		-	不能 `\` 反斜杠，考虑创建临时变量
	-	格式化字符串字面值可以拼接，但是一个替换字段不能拆分到多个字面值中
	-	格式化字符串不能用作文档字符串，即使其中没有包含表达式

-	`{}` 标识的格式表达式中
	-	`=`：求值后同时显式表达式文本、结果
		-	`{}` 内表达式前、后空白符同样被输出
		-	常用于调试
	-	`!`：标识转换符
		-	`!s`：对结果调用 `str()`
		-	`!r`：调用 `repr()`
		-	`!a`：调用 `ascii()`
	-	`:`：标识格式说明符，结果使用 `format()` 协议格式化
		-	格式说明符被传入表达式或转换结果的 `.__format__()` 方法
		-	格式说明符微语言同 `str.format()` 方法，省略则传入空字符串
		-	顶层格式说明符可以嵌套（一层）替换字段
			-	嵌套字段可以包含自身的转换字段、格式说明符

```python
name = "Fred"
print(f"he said his name is {name!r}")
print("he said his name is {repr(name)}")				# 二者等价
number = 1024
print(f"{number: #0x}")									# 数值类型格式说明符
today = datetime(yeat=2017, month=1, day=27)
print(f"{today: %B %d, %Y}")							# 日期类型格式说明符
width, precision = 4, 10
value = decimal.Deciaml("12.345")
print(f"result: {value: {width}.{precision}}")			# 顶层格式说明符嵌套替换字段
```

###	数字字面值

-	数字字面值
	-	数字字面值不包括正负号
		-	负数实际上是由单目运算符 `-` 和字面值组合而成
	-	没有专门复数字面值
		-	复数以一对浮点数表示
		-	取值范围同浮点数
	-	数字字面值可以使用下划线 `_` 将数码分组提高可读性
		-	确定数字大小时，字面值之间的下滑线被忽略
		-	下划线可以放在数码之间、基数说明符 `0x` 等之后
		-	浮点数中不能直接放在 `.` 后

####	整形数字字面值

```lex
integer      ::=  decinteger | bininteger | octinteger | hexinteger
decinteger   ::=  nonzerodigit (["_"] digit)* | "0"+ (["_"] "0")*
bininteger   ::=  "0" ("b" | "B") (["_"] bindigit)+
octinteger   ::=  "0" ("o" | "O") (["_"] octdigit)+
hexinteger   ::=  "0" ("x" | "X") (["_"] hexdigit)+
nonzerodigit ::=  "1"..."9"
digit        ::=  "0"..."9"
bindigit     ::=  "0" | "1"
octdigit     ::=  "0"..."7"
hexdigit     ::=  digit | "a"..."f" | "A"..."F"
```

-	整形数字字面值
	-	没有长度限制，只受限于内存

####	浮点数字字面值

```lex
floatnumber   ::=  pointfloat | exponentfloat
pointfloat    ::=  [digitpart] fraction | digitpart "."
exponentfloat ::=  (digitpart | pointfloat) exponent
digitpart     ::=  digit (["_"] digit)*
fraction      ::=  "." digitpart
exponent      ::=  ("e" | "E") ["+" | "-"] digitpart
```

-	浮点数字字面值
	-	整形数部分、指数部分解析时总以 10 为计数
	-	浮点数字面值允许范围依赖于具体实现

####	虚数字面值

```lex
imagnumber ::=  (floatnumber | digitpart) ("j" | "J")
```

-	序数字面值将生成实部为 `0.0` 的复数

###	运算符

```lex
+       -       *       **      /       //      %      @
<<      >>      &       |       ^       ~
<       >       <=      >=      ==      !=
```

###	分隔符

```lex
(       )       [       ]       {       }
,       :       .       ;       @       =       ->
+=      -=      *=      /=      //=     %=      @=
&=      |=      ^=      >>=     <<=     **=
```

-	以上列表中后半部分为**增强赋值操作符**
	-	在词法中作为分隔符，同时起运算作用

```lex
'       "       #       \
```

-	以上可打印 *ASCII* 字符
	-	作为其他 token 组成部分时有特殊意义
	-	或对词法分析器有特殊意义

```lex
$       ?
```

-	以上可打印 *ASCII* 字符不在 Python 词法中使用
	-	出现在字符串字面值、注释之外将**无条件引发错误**



