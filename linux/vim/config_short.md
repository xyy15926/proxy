---
title: *Vim* KeyMap、Abbr
categories:
  - Linux
  - Tool
  - Vi
tags:
  - Linux
  - Tool
  - Vi
  - Keymapper
  - Abbr
date: 2019-08-01 01:52:54
updated: 2021-11-04 10:40:19
toc: true
mathjax: true
comments: true
description: Vim KeyMapper CMD
---

##	键盘映射

> - <https://yianwillis.github.io/vimcdoc/doc/map.html>

```vimscripts
:<mode>map <mark> {lhs} {rhs}
```

> - 注意：映射后不能跟注释，vim会认为整行都是命令

###	映射工作模式

-	映射的工作模式可区分 6 种
	-	*normal* 模式：输入命令时
	-	*visual* 模式：可视区域高亮并输入命令时
	-	*select* 模式：类似可视模式，但键入的字符对选择区替换
	-	*operator-pending* 模式：操作符等待中
	-	*insert* 模式：包括替换模式
	-	*command-line* 模式：输入 `:`、`/` 命令时

###	映射命令设置

-	映射命令设置的模式如下，对应都有如下非递归、取消命令
	-	`<mode>noremap[!]`：非递归映射，即不会在其他映射中再次被展开
	-	`<mode>unmap[!]`：取消映射
	-	`<mode>mapclear[!]`：

	|命令|模式|
	|-----|-----|
	|`:map`|*normal*、*visual*、*select*、*operator-pending*|
	|`:nmap`|*normal*|
	|`:vmap`|*visual*、*select*|
	|`:smap`|*selection*|
	|`:xmap`|*visual*|
	|`:omap`|*operator-pending*|
	|`:map!`|*insert*、*command-line*|
	|`:imap`|*insert*|
	|`:lmap`|*insert*、*command-line*、*Lang-Arg*|
	|`:cmap`|*command-line*|
	|`:tmap`|终端作业|

###	特殊参数

-	映射特殊参数
	-	`<buffer>`：映射将局限于当前缓冲区
		-	优先级比全局映射高
		-	清除映射时同样需要添加参数
		-	可使用 `<leader>` 替代 `<localleader>` 可工作，但是不推荐
	-	`<nowait>`：存在较短映射时，失效以其作为前缀的较长映射
	-	`<silent>`：映射不在命令行上回显
	-	`<special>`：特殊键可以使用`<>`记法
	-	`<script>`：映射只使用通过以`<SID>`开头来定义的脚本局部映射来重映射优右值中的字符
	-	`<unique>`：若存在相同命令、缩写则定义失败
		-	定义局部映射时，同样会检查全局映射
	-	`<expr>`：映射的右值将被作为表达式被计算

-	特殊参数说明
	-	特殊参数的尖括号 `<>` 是本身具有的，必须紧跟命令后面
	-	有些特殊参数在取消映射时同样需注明

###	`omap`

> - 应用方法：*operator* （操作） + *operator-pending* （移动、范围选择）

-	预定义的 *operator-pending* 映射如 `w`、`aw`、`i(`、`t,`

	|按键	|操作			|移动			|
	|-------|---------------|---------------|
	|dw		|删除(delete)	|到下一个单词	|
	|ci(	|修改(change)	|在括号内		|
	|yt,	|复制			|到逗号前		|

-	自定义的 *operator-pending* 映射则需要

	-	选取一定范围：可同时指定开头、结尾（一般通过进入 *visual* 模式下选择范围）

		```vimscripts
		" 下个括号内内容
		onoremap in( :<c-u>normal! f(vi(<cr>
		" 当前括号内容
		onoremap il( :<c-u>normal! f)vi(<cr>`
		" 选取使用 `===` 标记 markdown 标题
		onoremap ih :<c-u>execute "normal! ?^==\\+$\r:nohlsearch\rkvg_"<cr>
		onoremap ah :<c-u>execute "normal! ?^==\\+$\r:nohlsearch\rg_vk0"<cr>
		```
	-	指定光标位置：光标当前位置为开头、指定位置为结尾

		```vimscripts
		" 移动至 `return` 前一行
		onoremap b /return<cr>
		```

###	*leaders*、*localleader*

*leader*、*localleader*：作为“前缀”的不常用的按键，后接其他字符作为整体映射

-	用途
	-	避免覆盖太多按键原始功能 
	-	约定俗成的规范，容易理解
	-	方便更改 `<leader>`、`<localleader>` 作为前缀设置
		-	`<leader>`：对全局映射而设置的映射的前缀
		-	`<localleader>`：只对某类（个）文件而设置的映射的前缀
	-	`<leader>` 和 `<localleader>` 除了设置不同以外，没有太大区别，应用场合时约定规范，不是强制性的

-	`<leader>`、`<localleader>` 设置

	```vimscripts
	:let mapleader = "-"
	:nnoremap <leader>d dd
	:let maplocalleader = "\\"
	:nnoremap <buffer> <localleader>c I#<esc>
	```

	-	*vim* 会对 `mapleader`、`maplocalleader` 进行特殊的处理，不是简单的声明

##	*Abbreviations* 缩写

-	`iabbrev`：紧跟缩写输入非关键字后，缩写会替换为相应的完整字符串
	-	相较于映射
		-	`iabbrev` 用于 *insert*、*replace*、*command-line* 模式
		-	`iabbrev` 会注意缩写前后的字符，只在需要的时候替换
	-	`iabbrev` 同样支持特殊参数
		-	`<buffer>`：仅限本地缓冲区

```vimscripts
" 纠错
iabbrev waht what
" 简化输入
iabbrev @@ xyy15926@gmail.com
" 替换 `----` 为前个单词
iabbrev <buffer> ---- &mdash
" 替换 `return` 为 null
iabbrev <buffer> return nopenopenope
```

> - `:set iskeyword?` 即可查看关键字字符



