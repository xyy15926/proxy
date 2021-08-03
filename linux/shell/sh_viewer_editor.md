---
title: 文件、目录
tags:
  - Linux
  - Shell
categories:
  - Linux
  - Shell
  - Editor
  - Viewer
date: 2019-07-31 21:10:52
updated: 2021-07-30 10:49:09
toc: true
mathjax: true
comments: true
description: 文件、目录
---

##	显示文本文件

###	`cat`

> - `cat`：显示文本文件内容
> - `zcat`：查看压缩文件内容

###	`more`

> - `more`：单向分页显示文本文件
> - `zmore`：单向分页显式压缩文本文件

###	`less`

> - `less`：双向分页显示文本文件内容
> - `zless`：双向分页显式压缩文件内容

###	`head`

`head`：显示文件指定前若干行

###	`tail`

`tail`：实现文件指定后若干行

###	`nl`

`nl`：显示文件行号、内容

##	文件处理

###	`sort`

对文件中数据排序

###	`uniq`

删除文件中重复行

###	`cut`

从文件的每行中输出之一的字节、字符、字段

###	`diff`

逐行比较两个文本文件

###	`diff3`

逐行比较三个文件

###	`cmp`

按字节比较两个文件

###	`tr`

从标准输入中替换、缩减、删除字符

###	`split`

将输入文件分割成固定大小的块

###	`tee`

将标准输入复制到指定温婉

###	`expand`

将文件中`tab`转换为空格输出到标准输出

```c
$ expand -n 4 file_name
```

###	`nano`

###	`awk`

一门模式匹配的编程语言

-	主要功能是匹配文本并处理
-	同时还有一些编程语言才有的语法：函数、分支循环语句、变量
	等等
-	使用*awk*可以
	-	将文本文件视为字段、记录组成的文本数据库
	-	操作文本数据库时能够使用变量
	-	能够使用数学运算和字符串操作
	-	能够使用常见地编程结构，如：条件、分支循环
	-	能够格式化输出
	-	能够自定以函数
	-	能够在*awk*脚本中执行linux命令
	-	能够处理linux命令的输出结果

####	命令行语法

```shell
$ awk [-F ERE] [-v assignment] ... program [argument...]
$ awk [-F ERE] -f progfile ... [-v assignment] ... [argument ...]
```

###	`sed`

`sed`：非交互式、面向字符流的编辑器

-	`sed`也是默认从`stdin`读取输入、输出至`stdout`，除非
	参数`filename`被指定，会从指定文件获取输入，但是注意
	sed是面向字符流的编辑器，所以输入、输出文件不能是同一个

-	sed按行处理文本数据，每次处理一行在行尾添加换行符

```shell
$ sed [-hnV] [-e<script>][-f<script-file>][infile]
```

####	参数

-	`-e<script>/--expression=<script>`：以指定script
	处理infile（默认参数）
	-	默认不带参数即为`-e`

-	`-f<script-file>/--file=<script-file>`：以指定的script
	文件处理输入文本文件
	-	文件内容为sed的动作

-	`-i`：直接修改原文件

-	`-n/--quiet`：仅显示script处理后结果

-	`-h/--help`：帮助

-	`-V/--version`：版本信息

####	动作

-	`[n]a\string`：行添加，在`n`行后添加新行`string`
-	`[n]i\string`：行插入
-	`[n]c\string`：行替换
-	`[n,m]d`：删除，删除`n-m`行
-	`[start[,end]]p`：打印数据
-	`[start[,end]]s/expr/ctt[/g]`：正则替换

####	高级语法

####	示例

```md
$ sed '2anewline' ka.file
$ sed '2a newline' ka.file
$ sed 2anewline ka.file
$ sed 2a newline ka.file
	# 在第2行添加新行`newline`

$ sed 2,$d ka.file
	# 删除2至最后行

$ sed 2s/old/new ka.file
	# 替换第2行的`old`为`new`

$ nl ka.file | sed 7,9p
	# 打印7-9行

$ sed ":a;N;s/\n//g;ta" a.txt
	# 替换换行符
```

##	查找字符串、文件

###	`grep`

查找符合条件的字符串

###	`egrep`

在每个文件或标准输入中查找模式

###	`find`

列出文件系统内符合条件的文件

###	`whereis`

插卡指定文件、命令、手册页位置

###	`whatis`

在whatis数据库中搜索特定命令

###	`which`

显示可执行命令路径

###	`type`

输出命令信息

-	可以用于判断命令是否为内置命令

