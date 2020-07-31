---
title: Linux文本工具
tags:
  - Linux
  - Shell
  - Command
categories:
  - Linux
  - Shell Command
date: 2019-07-31 21:10:52
updated: 2019-02-17 11:57:07
toc: true
mathjax: true
comments: true
description: Linux文本工具
---

##	Awk

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

###	命令行语法

```shell
$ awk [-F ERE] [-v assignment] ... program [argument...]
$ awk [-F ERE] -f progfile ... [-v assignment] ... [argument ...]
```

##	Sed

A non-interactive stream-oriented editor，非交互式、面向
字符流的编辑器

```shell
$ sed [options] script filename
```

-	`sed`也是默认从`stdin`读取输入、输出至`stdout`，除非
	参数`filename`被指定，会从指定文件获取输入，但是注意
	sed是面向字符流的编辑器，所以输入、输出文件不能是同一个

##	Grep



