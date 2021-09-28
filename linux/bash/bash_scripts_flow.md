---
title: Bash 执行控制
categories:
  - Linux
  - Bash
tags:
  - Linux
  - Bash
  - Shell
  - Grammer
date: 2019-08-01 01:37:27
updated: 2021-08-24 10:40:43
toc: true
mathjax: true
comments: true
description: Shell执行控制
---

##	分支

###	`true`、`false`

-	`true`：表示操作成功，Bash 内建命令
-	`false`：表示操作失败，Bash 内建命令

###	`if`

```sh
if <if-cmd>; then
	<do-cmd>
[elif <if-cmd>; then
	<do-cmd>
[else
	<do-cmd>]
fi
```

-	`if-cmd` 命令表达式的返回结果（退出状态）用于决定分支执行
	-	命令返回（退出状态）为 `0` 时，分支被执行
	-	可以是 `;` 分隔的多条命令，多条命令都会被执行，但只有最后一条结果影响分支执行

-	`if-cmd` 一般为 `test` 命令（可以是任何命令），有如下三种略有不同的等价形式
	-	`test <expr>`
	-	`[ <expr> ]`
	-	`[[ <expr> ]]`

-	`if-cmd` 其他常用的命令包括
	-	`true`、`false`：恒真、恒假
	-	`(())`、`let`：计算表达式
		-	判断依据是返回值，与 `if` 配合刚好和计算结果逻辑相同

###	`case`

```shell
case <expr> in
	<ptn1> | <ptn1_alter>)
		<do-cmd> ;;
	<ptn2>)
		<do-cmd> ;;
	*)
		<do-cmd> ;;
esac
```

-	`expr`：待匹配表达式
-	`ptn`：分支条件
	-	分支条件允许使用通配符（基本支持所有 *Globbing* 通配符扩展）
		-	`*`、`?`、`|`
		-	`[<start>-<end>]`
		-	`[[:<class>:]]`
-	分支语句结束符
	-	`;;`：终止模式，匹配成功之后即退出 `case` 模块
	-	`;;&`：非终止模式，匹配成功之后继续匹配剩余分支

##	`test`、`[]`、`[[]]`

-	`test <expr>` 中
	-	表达式 `expr` 为真时，`test` 执行成功，返回 0
	-	表达式 `expr` 为假时，`test` 执行失败，返回 1

-	`[` 实际上是 `test` 命令的简写，`]` 指示表达式结束
	-	所以 `[ <expr> ]` 中表达式 `expr` 和 `[]` 必须有空格
	-	比较总结
		-	`=`、`\>`、`/<` 比较字符串
		-	连词线参数比较整数值、文件
		-	连词线参数进行逻辑运算

-	`[[]]` 是 `[]` 的逻辑判断增强版
	-	其中不支持文件名扩展（所以允许 `*`、`?` 等符号用于正则）
	-	支持 `=~` 比较符正则匹配
	-	比较总结
		-	`=`、`>`、`<` 比较字符串
		-	连词线参数比较整数值、文件
		-	`&&`、`||` 进行逻辑运算

-	注意
	-	参数为变量时考虑用 `""` 扩起
		-	避免 Shell 扩展影响；添加相同字符
		-	避免空值影响结果
			-	`-e` 等单目检验标志在参数为空时退出值总为 0，影响逻辑
			-	`-a` 等双目检验标志在参数为空时报错
	-	`[]`、`test` 是 *POSIX* Shell 协议要求，所有 Shell 均支持
		-	`[[]]` 是 *Bash* 的自定义命令

###	与、或、非逻辑运算

-	`test` 内、`[]`逻辑运算
	-	`-a`：`AND` 与
	-	`-o`：`OR` 或
	-	`!`：`NOT` 非

-	`test` 命令（和其他命令）间、`[[]]` 内逻辑运算
	-	`&&`：`AND` 与
	-	`||`：`OR` 或（短路运算）
	-	`!`：`NOT` 非

	```shell
	[[ $a != 0 && $b == 1 ]]
	[ $a -eq 0 ] && [ $b -eq 1 ]
	[ $a -eq 0 -a $b -eq 1]
	```

-	说明
	-	可以使用括号明确范围，但括号需要转义、空格分隔

	```sh
	if [ ! \( $INT -ge $MIN_VAL -a $INT -le $MAX_VAL \) ]; then
		echo "$INT is outside $MIN_VAL to $MAX_VAL"
	else
		echo "INT is in range"
	fi
	```

###	文件检验标志

-	文件是否存在
	-	`-a`/`-e`：文件是否存在

-	文件类型
	-	`-f`：文件为一般文件（非目录、设备）
	-	`-s`：文件大小非 0
	-	`-d`：目标为目录
	-	`-b`：文件为块设备（软盘、光驱）
	-	`-c`：文件为流设备（键盘、modem、声卡）
	-	`-p`：文件为管道
	-	`-h`/`L`：目标为符号链接
	-	`-S`：目标为 Socket

-	文件权限
	-	`-r`：目标是否可读权限
	-	`-w`：目标是否可写权限
	-	`-x`：目标是否可执行权限
	-	`-g`：目标是否设置有 *set-group-id*
	-	`-u`：目标是否设置有 *set-user-id*
	-	`-k`：目标是否设置 *sticky bit*
	-	`-O`：用户是否是文件所有者
	-	`-G`：用户是否属于文件所有组

-	文件状态
	-	`-N`：文件从上次读取到现在为止是否被修改

-	文件描述符
	-	`-t`：为文件描述符被关联到终端
		-	一般用于检测脚本的stdin是否来自终端`[ -t 0 ]`，或
			输出是否输出至终端`[ -t 1 ]`

-	文件比较
	-	`<f1> -nt <f2>`：文件 `f1` 比 `f2` 新，或 `f1` 存在而 `f2` 不存在
	-	`<f1> -ot <f2>`：文件 `f1` 比 `f2` 旧，或 `f2` 存在而 `f2` 不存在
	-	`<f1> -et <f2>`：文件 `f1`、`f2` 指向相同文件（即硬链接）

###	字符串检验

-	`-n`：字符串不为空
	-	`test <str>` 可以直接判断字符串是否为空
-	`-z`：字符串为空
-	`<str1> = <str2>`：字符串 `str1`、`str2` 相同
-	`<str1> == <str2>`：字符串 `str1`、`str2` 相同（*Bash* 独有）
-	`<str1> != <str2>`：字符串 `str1`、`str2` 不同
-	`<str1> > <str2>`：字符串 `str1` 按字典序排在 `str2` 后
	-	在 `[]`、`test` 中使用需转义 `\<`，否则被视为重定向
-	`<str1> < <str2>`：字符串 `str1` 按字典序排在 `str2` 前

> - 注意：`test 0` 中 `0` 是作为字符串（非空）被检验，所以退出值是 `0`（执行成功），需要判断整数，应使用整数算术命令 `((0))`

####	`[[]]` 正则判断

-	`[[ <str> =~ <regex> ]]`

	```sh
	if [[ "$INT"]] =~ "^-?[0-9]+$" ]]; then
		echo "INT is an integer."
		exit 0
	else
		echo "INT is not an integer." > &2
		exit 1
	fi
	```

###	整数判断

-	`<int1> -eq <int2>`：整数 `int1`、`int2` 相等
-	`<int1> -ne <int2>`：整数 `int1`、`int2` 不等
-	`<int1> -le <int2>`：整数 `int1` 小于或等于 `int2`
-	`<int1> -lt <int2>`：整数 `int1` 小于 `int2`
-	`<int1> -ge <int2>`：整数 `int1` 大于或等于 `int2`
-	`<int1> -gt <int2>`：整数 `int1` 大于 `int2`



-	`test <expr>`：bash内部命令，判断表达式是否为真

	-	其后只能有一个表达式判断，无法使用逻辑与、或连接

-	`[ <expr> ]`：基本等同于`test`

	-	整数比较：`-ne`、`-eq`、`-gt`、`-lt`
	-	字符串比较：转义形式大、小号`\<`、`/>`
	-	逻辑表达式连接：`-o`（或）、`-a`（与）、`!`（非）
		（不能使用`||`（或）、`&&`（与））

	```shell
	if [ $? == 0 ]
	if test $? == 0
		# 二者等价
	```

	> - 注意内部**两边要有空格**
	> - 事实上是`[`等同于`test`，`]`表示关闭条件判断，新版
		bash强制要求闭合

-	`[[ <expr> ]]`条件测试

	-	整数比较：`!=`、`=`、`<`、`>`
	-	字符串：支持模式匹配（此时右侧模式时不能加引号）
	-	逻辑表示式连接：`||`（或）、`&&`（与）、`!`（非）


	> - `[[`：bash关键字，bash将其视为单独元素并返回退出
		状态码

##	循环

###	`while`

```shell
while <while-cmd>; do
	<do-cmd>
done
```

-	同 `if-cmd`，`while-cmd` 命令表达式的返回结果（退出状态）用于决定循环体执行
	-	命令返回（退出状态）为 `0` 时，判断成功，循环体被执行
	-	可以是 `;` 分隔的多条命令，多条命令都会被执行，但只有最后一条结果影响分支执行

-	同 `if-cmd`，`while-cmd` 一般为 `test` 命令（可以是任何命令），有如下三种略有不同的等价形式
	-	`test <expr>`
	-	`[ <expr> ]`
	-	`[[ <expr> ]]`

####	`until`

```shell
until <until-cmd>; do
	<do-cmd>
done
```

-	`until-cmd` 命令表达式的返回结果（退出状态）用于决定循环体执行
	-	与 `if-cmd`、`while-cmd` 相反，命令返回（退出状态）为 `0` 时，判断失败，循环体终止执行
	-	可以是 `;` 分隔的多条命令，多条命令都会被执行，但只有最后一条结果影响分支执行

-	同 `if-cmd`，`until-cmd` 一般为 `test` 命令（可以是任何命令），有如下三种略有不同的等价形式
	-	`test <expr>`
	-	`[ <expr> ]`
	-	`[[ <expr> ]]`

###	`for`

####	`for...in...` - ListStyle

```sh
for <var> [ [ in [<var-list>] ] ];]  do
	<do-cmd>
done
```

-	迭代风格 `for`
	-	扩展  `var-list` 成为列表（分词结果）
		-	`in <var-list>` 缺省为 `in $@`，即默认迭代位置参数
	-	将 `var-list` 中成员赋值给 `var`
	-	对 `var` 执行循环体

-	说明
	-	退出状态由最后执行命令决定
		-	`var-list` 为空时，退出值为 0

-	常用

	-	结合文件名扩展，迭代类型文件

		```sh
		for <var> in </path/to/file_ptn>; do
			<do-cmd>
		done
		```

	-	结合花括号扩展、`seq`，迭代序列

		```shell
		# `seq` 命令生成序列
		for i in $(seq <start> <end>); do
			<do-cmd>
		done

		# 花括号扩展为序列
		for i in {<start>..<end>..<step>}; do
			<do-cmd>
		done
		```

####	`for ((...))` - CStyle

```sh
for (( <init-expr>; <check-expr>; <loop-expr> )); do
	<do-cmd>
done
```

-	*C* 风格迭代
	-	计算算术表达式 `init-expr`
	-	检查算术表达式 `check-expr`，直至其值为 0 则终止循环
	-	每次检查 `check-expr` 都执行一次 `loop-expr`

###	`select`

```sh
select <var> [ [ in [<var-list>] ] ];]  do
	<do-cmd>
done
```

-	`select` 在标准输出中输出选项菜单，并执行循环体
	-	将 `var-list` 成员打印至标准错误输出，`var-list` 缺省为 `"$@"`
	-	从标准输入获取用户选项，存储至 `var` 中，用户实际输入存储在 `$REPLY` 中
	-	为用户每个输入执行循环体
	-	需要显式 `break` 才能退出

-	说明
	-	`select` 常与 `case` 语句搭配使用

###	`break`、`continue`


-	`break <n>`：跳出 `n` 层循环体
	-	`n` 缺省为 1，即跳出一层循环体
-	`continue [n]`：跳过 `n` 层循环体在其之后的语句，回到循环开头
	-	`n` 缺省为 1，即跳过一次循环体


##	其他关键字

-	`exit <val>`：退出程序，后跟退出值
-	`return <val>`：结束函数，并将 `val` 返回给调用者
	-	`val` 缺省为函数中最后一条命令退出值
-	`:`：内建空指令，返回值为0
	-	常用于占位

