---
title: Vim 配置
categories:
  - Linux
  - Tool
  - Vi
tags:
  - Linux
  - Tool
  - Vi
  - Configuration
date: 2019-08-01 01:53:29
updated: 2021-08-04 19:43:08
toc: true
mathjax: true
comments: true
description: vimscriptss基础--Setting
---

##	打印信息

-	`:echo`：打印信息，但是信息不会保存
-	`:echom`：打印信息会保存在`:messages`中
-	`:messages`：查看`:echom`保存的信息

##	设置选项

-	设置选项方式
	-	命令行一次设置多个选项：`:set number numberwidth=6`
	-	本地缓冲区设置：`:setlocal nonumber`

-	设置 *bool* 选项
	-	`:set <name>`：打开选项
	-	`:set no<name>`：关闭选项
	-	`:set <name>!`：切换选项
	-	`:set <name>?`：查看选项值（返回<name>或no<name>）

-	设置键值选项
	-	`:set <name>=value`：设置选项值
	-	`:set <name>?`：查看选项值
	-	键值选项支持运算：`+=` 等

###	 *statusline* 状态栏设置

-	状态栏代码通用格式：`%-0{minwid}.{maxwid}{item}`

	-	`-`：左对齐
	-	`0`：使用"0"填充
	-	`%=`：切换到**状态栏**右侧

	-	`%f`：文件名
	-	`%F`：完整路径文件名
	-	`%y`：文件类型（[text]，[python]）
	-	`%Y`：文件类型（TEXT，PYTHON）

	-	`%l`：当前行号
	-	`%L`：总行数
	-	`%v/%V`：列号
	-	`%c`：字符列号
	-	`%p`：文件位置百分比

	-	`%n`：buffer number
	-	`%{&ff}`：文件格式（DOS、UNIX）
	-	`%b`：当前字符ACSII码
	-	`%B`：当前字符16进制值
	-	`%m`：modified flag（[+]，[-]表示不可修改）

-	设置状态栏显式格式，既可以一行完成配置，也可以分开配置

	```vimscripts
	set statusline=%f\ -\ filetype:\ %y
	set statusline=%f
	set statusline+=%=
	set statusline+=%l
	set statusline+=/
	set statusline+=%L
	```
	-	中间空格需要用“\“转义，“%%”转义“%”

-	`laststatus`：设置状态栏显示模式
	-	`1`：默认值，两个以上窗口才显示
	-	`2`：一直显示

###	折叠设置

-	`foldmethod`
	-	`manual`：手动折叠选中的行（默认 `zf` 触发）
	-	`marker`：`{{{` 到 `}}}` 标记待折叠行（默认 `za` 触发）
	-	`indent`：折叠缩进
	-	`syntax`：语法折叠
-	`foldlevel=<num>`：设置 *indent* 折叠起始水平（`zm` 触发），即从 `<num>` 水平开始尝试折叠
-	`foldlevelstart=<num>`：设置文件打开时默认折叠水平
	-	`-1`：初始不折叠
-	`foldcolumn=<num>`：用 `<num>` 行表示可可折叠状态

###	一些常用关键字

-	`iskeyword=@,_,48-57,192_255`：指定关键字
	-	下划线
	-	*ASCII* 码位在 48-57 之间的字符（`0-9`）、192-255之间的字符

-	`conceallevel=0`：隐藏等级
	-	`1`
	-	`2`

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
	-	特殊参数的尖括号`<>`是本身具有的，必须紧跟命令后面
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

##	*Autocmd* 自动命令

###	`autocmd` 使用

-	`autocmd` 注意事项
	-	同时监听多个事件，使用 `,` 分隔，中间不能有空格
	-	一般同时监听 `bufnewfile`、`bufread`，这样打开文件时无论文件是否存在都会执行命令
	-	所有事件后面都需要注明适用场景，可用`*`表示全部场景，中间也不能有空格
	-	`autocmd` 是定义命令，不是执行命令
		-	每次执行都会定义命令，而*vim* 不会忽略重复定义
		-	如：`:autocmd bufwrite * :sleep 200m`，每次执行时都会重复定义命令

-	缓冲区事件

	```vimscriptss
	autocmd bufnewfile * :write
	autocmd bufnewfile *.txt :write
	autocmd bufwritepre *.html :normal gg=g
	autocdm bufnewfile,bufread *.html setlocal nowrap
	```

-	*filetype* 事件（*vim* 设置缓冲区 *filetype* 时触发）

	```vimscriptss
	autocmd filetype javascript nnoremap <buffer> <localleader>c i//<esc>
	autocmd filetype python nnoremap <buffer> <localleader>c i#<esc>
	autocmd filetype javascript :iabbrev <buffer> iff if ()<left>
	autocmd filetype python :iabbrev <buffer> iff if:<left>
	```

###	`augroup` 自动命令组

-	自动命令组

	```vimscripts
	augroup cmdgroup
		autocmd bufwrite * :echom "foo"
		autocmd bufwrite * :echom "bar"
	augroup end
	```

-	注意事项
	-	类似 `autocmd`，*vim* 不会忽略重复定义，但是可以通过 `:autocmd!` 清除一个组

		:augroup cmdgroup
		:	autocmd!
		:	autocmd bufwrite * :echom "foo"
		:	autocmd bufwrite * :echom "bar"
		:augroup end

##	Vim安装

###	安装选项

```shell
$ ./configure --with-features=huge\
	--enable-multibyte \
	--enable-python3interp \
	--with-python3-config-dir=/usr/lib64/python3.4/config-3.4m/ \
	--enable-pythoninterp \
	--with-python-config-dir=/usr/lib64/python2.7/config/ \
	--prefix=/usr/local
	--enable-cscope
```
-	按照以上命令配置，编译出的Vim版本中是**动态**支持
	`+python/dyn`和 `+python3/dyn`
	
-	此时Vim看似有python支持，但是在Vim内部
	`:echo has("python")`和`:echo has("python3")`都返回`0`

-	之后无意中尝试去掉对`python`的支持，编译出来的Vim就是
	可用的`python3`，不直到为啥
