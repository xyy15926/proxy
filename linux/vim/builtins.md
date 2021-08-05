---
title: Vim 内建函数、变量
categories:
  - Linux
  - Tool
  - Vi
tags:
  - Linux
  - Tool
  - Vi
date: 2019-05-11 09:55:22
updated: 2021-08-04 10:51:37
toc: true
mathjax: true
comments: true
description: Vim内建函数、变量
---

##	文件、路径相关函数

-	`expand(option)`：根据参数返回当前文件相关信息
-	`fnamemodify(file_name, option)`：返回当前文件夹下文件
	信息
-	`globpath(dir, type)`：返回的`dir`下符合`type`的文件
	列表值字符串，使用`,`分隔，`type`为`**`时将递归的列出
	文件夹及文件

##	特殊变量

command-line模式的特殊变量，在执行命令前会将其替换为相应的
变量

-	`<cword>`：光标处单词
-	`<cWORD>`：光标处单词大写形式

##	寄存器

###	寄存器相关快捷键、命令

-	`<c-r><reg>`：insert模式下直接输入`<reg>`中的值

###	一般寄存器

###	Readonly Register

###	Expression Register（`"=`）

`"=`实际上并不是一个寄存器，这是使用命令表达式的一种方法，
按下`=`之后，光标会移动到命令行，此时可以输入任何表达式，
（不只有`"=`才会激活命令行，`<c-m>"`也能激活）
输入表达式之后

-	按下`<esc>`，表达式值被丢弃
-	按下`<cr>`，表达式值计算后存入`"=`中
	```vim
	:nnoremap time "=strftime("%c")<cr>p
	:inoremap time <c-r>strftime("%c")<cr>
	```

之后`:put`或者按下`p`将粘贴`"=`中的值

>	寄存器中的值一定是字符串，如果是其他类型变量，会被强制
	转换之后存入`"=`寄存器中

##	Vim特殊

###	换行

-	`\0`：空转义序列（ASCII码位0）`<Nul>`
	-	`<c-v> 000`：输入`<Nul>`
	-	Vim在内存中使用`<NL>`存储`<Nul>`，在读、写文件时即
		发生转换

	> - Vi无法处理`<Nul>`，应该是为了兼容Vi

-	`\n`：换行转义序列`<NL>`
	-	`<c-v><c-j>`：输入`<NL>`，会被替换为输入`<Nul>`，
		等同于`<c-v> 000`
	-	在搜索表达式中：字面意义的`newline`序列被匹配
	-	在替换表达式中：在内部被替换为`<Nul>`被输入，即
		**不再表示newline**

-	`\r`：回车转义序列`<CR>`
	-	被Vim视为为换行，可在替换表达中表示`<NL>`
	-	`<c-v><c-j>`：输入`<CR>`字符本身

> - vim在内存换行应该同一使用`<CR>`，在读、写时，根据当前
	`fileformat`设置自动转换换行字符（序列）


