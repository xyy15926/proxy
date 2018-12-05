#	Vim内建函数、变量

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

