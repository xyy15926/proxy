#	vimscripts基础

##	打印信息

-	`:echo`：打印信息，但是信息不会保存
-	`:echom`：打印信息会保存在`:messages`中
-	`:messages`：查看`:echom`保存的信息

##	设置选项

-	命令行一次设置多个选项：`:set number numberwidth=6`
-	本地缓冲区设置：`:setlocal nonumber`

###	bool选项

-	`:set <name>`：打开选项
-	`:set no<name>`：关闭选项
-	`:set <name>!`：切换选项
-	`:set <name>?`：查看选项值（返回<name>或no<name>）

###	键值选项

-	`:set <name>=value`：设置选项值
-	`:set <name>?`：查看选项值

##	键盘映射

###	映射设置格式

-	基本映射`:map`：`:map - dd`，`:map <space> viw`，
	`:map <c-d> dd`
-	非递归映射`:*noremap`：`:nnoremap - dd`
-	解除映射：`:*unmap -a`
-	本地缓冲区映射：`:nnoremap <buffer> <localleader>x dd`
	-	<buffer>：只有定义此映射的缓冲区有效
	-	其优先级比“普通（无<buffer>）”高，即使先定义
	-	这个命令中使用<leader>替代<localleader>可工作，但是
		不推荐，既不利于理解，也可能覆盖别人的设置

**注意：映射后不能跟注释，vim会认为整行都是命令**

###	映射模式

-	`:map`：在所有模式下工作
-	`:nmap`：仅normal模式下有效

		:nnoremap <leader>sv :source $MYVIMRC<cr>
		:nnoremap <leader>" viw<esc>a"<esc>hbi"<esc>lel
		:vnoremap <leader>" :'<I"<esc>:'>I"<esc>

-	`:vmap`：仅visual模式下有效
-	`:imap`：仅insert模式下有效

		:inoremap <c-d> <esc>ddi
		:inoremap <c-u> <esc>viwUea
		:inoremap <esc> <nop>（no operation）

-	`:omap`：Operator-Pending映射（movement映射）

		:onoremap p i(（将p映射为“括号内”）

	-	operator（操作）+ movement（移动）

		|按键		|操作			|移动			|
		|-----------|---------------|---------------|
		|dw			|删除(delete)	|到下一个单词	|
		|ci(		|修改(change)	|在括号内		|
		|yt,		|复制			|到逗号			|

		movement在normal模式下是移动（大部分，"i("不是），在
		visual模式下是**选取**将movemnt看作是visual模式下的
		选取比较好，虽然这些快捷键都是normal模式下使用的

	-	movement不仅仅可以是预定义“w,i(,t”等，还可以是任何
		涉及光标移动的命令，即**visual模式下光标选取的区域**

		-	`:onoremap b /return<cr>`：不改变光标起始位置，
			映射为“到’return‘前一行“，查找导致光标移动选取

		-	`:onoremap in( :<c-u>normal! f(vi(<cr>`：
			映射为“下一个括号内容”，这里使用了`v`显式进入
			visual，然后选取内容

		-	`:onoremap il( :<c-u>normal! f)vi(<cr>`：
			“当前括号内容”

		-	`:onoremap ih :<c-u>execute 
				"normal! ?^==\\+$\r:nohlsearch\rkvg_"<cr>`：
			转义后相当于执行
			`:normal! ?^==\+$<cr>:nohlsearch<cr>kvg_`，
			选取使用`===`标记markdown标题

		-	`:onoremap ah :<c-u>execute 
				"normal! ?^==\\+$\r:nohlsearch\rg_vk0"<cr>`

		>	`<c-u>`、`:normal!`、`execute`以及正则表达式说明
			见其他文件

-	好像还有各paste模式下，:map映射也没用

###	Leaders

leader：作为“前缀”的不常用的按键，后接其他字符作为整体映射，
避免覆盖太多按键原始功能 

> - 方便更改leader设置
> - 约定俗成的规范，容易理解
> - `<leader>`和`<localleader>`除了设置不同以外，没有太大的
	区别，应用场合时约定规范，不是强制性的，也可以用
	`<localleader>`设置为全局的快捷键（不用`<buffer>`）

-	设置`<leader>`

		:let mapleader = "-"
		:nnoremap <leader>d dd

	这个变量vim会进行特殊的处理，不是简单的声明

-	设置`<localleader>`，用于只对某类（个）文件而设置的映射

		:let maplocalleader = "\\"
		:nnoremap <buffer> <localleader>c I#<esc>

##	abbreviations（缩写）

紧跟`abbreviation`输入”non-keyword character“后，vim会替换为
相应的完整字符串，相较于`map`

-	abbreviation用于insert、replace、command模式
-	abbreviation会注意`iabbrev`前后的字符，只在需要的时候替换
	（`iabbrev`后跟“no-keyword character时”）

	> - “non-keyword character”指不在iskeyword选项中的字符
	> - `:set iskeyword?`即可查看包含的字符
	> - 默认`iskeyword=@,_,48-57,192_255`
		>> - 所有字母，包括大小写
		>> - 下划线”_“
		>> - ascii码位在48-57之间的字符0-9
		>> - ascii码位在192-255之间的字符

-	全局缩写

		:iabbrev waht what（纠错）
		:iabbrev @@ xyy15926@gmail.com（简化输入）

-	本地缓冲区缩写

		:iabbrev <buffer> ---- &mdash（“----”替换为前个单词）
		:iabbrev <buffer> return nopenopenope（替换为null）

##	autocmd（自动命令）

-	autocmd使用格式

	-	缓冲区事件
		```vimscripts
		:autocmd bufnewfile * :write
		:autocmd bufnewfile *.txt :write
		:autocmd bufwritepre *.html :normal gg=g
		:autocdm bufnewfile,bufread *.html setlocal nowrap
		```

	-	filetype事件（vim设置缓冲区filetype时触发）

			:autocmd filetype javascript nnoremap <buffer> <localleader>c i//<esc>
			:autocmd filetype python nnoremap <buffer> <localleader>c i#<esc>
			:autocmd filetype javascript :iabbrev <buffer> iff if ()<left>
			:autocmd filetype python :iabbrev <buffer> iff if:<left>

	>-	同时监听多个事件，使用“,”分隔，中间不能有空格
	>-	一般bufnewfile、bufread事件同时监听，这样打开文件时
		无论文件是否存在都会执行命令
	> - 所有事件后面都需要注明适用场景，如果不知道注明什么，
		用`*`表示全部场景，中间也不能有空格

-	自动命令组

		:augroup cmdgroup
		:	autocmd bufwrite * :echom "foo"
		:	autocmd bufwrite * :echom "bar"
		:augroup end

	vim不会替换命令，即使完全一样
	`:autocmd bufwrite * :sleep 200m`
	这个命令如果在`~/.vimrc`中，每次`:source $myvimrc`时都会
	定义 这样一个命令，多次之后每次写入就会很大延迟

	和autocmd一样，vim不会替换自动命令组
	但是可以:autocmd!清除一个组

		:augroup cmdgroup
		:	autocmd!
		:	autocmd bufwrite * :echom "foo"
		:	autocmd bufwrite * :echom "bar"
		:augroup end

	如此，每次都会清除自动命令组，相当于替换

##	界面样式

###	状态栏（statusline）

设置状态栏显式格式，既可以一行完成配置，也可以分开配置

	:set statusline=%f\ -\ filetype:\ %y

	:set statusline=%f
	:set statusline+=%=
	:set statusline+=%l
	:set statusline+=/
	:set statusline+=%L

>	中间空格需要用“\“转义，“%%”转义“%”

状态栏代码通用格式：`%-0{minwid}.{maxwid}{item}`

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

`:set laststatus=2`：设置状态栏一直显示，默认值是1，两个
以上窗口才显示


###	vim代码折叠

-	`set foldmethod=marker`：normal模式下`za`会折叠
	`{{{`到`}}}`之间的行（光标位于其中）

	>	这种折叠只应该用于vim配置文件，否则对非vim用户不友好，
		而非vim用户不会处理vim配置文件

		" vimscript file settings -----{{{
		augroup cmdgp
			autocmd!
			autocmd filetype vim setlocal foldmethod=marker
		augroup end
		" }}}

-	`set foldlevelstart=0`：设置默认折叠所有添加折叠注释
-	`set foldleve=num`：折叠比num level高的可折叠

