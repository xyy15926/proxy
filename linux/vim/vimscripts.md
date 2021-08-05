---
title: Vimscripts 编程
categories:
  - Linux
  - Tool
  - Vi
tags:
  - Linux
  - Tool
  - Vi
  - Grammer
  - VimScripts
date: 2019-03-21 17:27:37
updated: 2021-08-04 10:52:55
toc: true
mathjax: true
comments: true
description: vimscripts函数编程
---

##	变量

###	普通变量

创建变量、变量赋值都需要用到`let`关键字

```vim
let foo = "bar"
echo foo
let foo = "foo"
echo foo
```

####	数字

-	`Number`32位带符号整形（整形之间除法同c）

	-	`:echo 0xef`：16进制
	-	`:echo 017`：8进制（鉴于以下，不建议使用）
	-	`:echo 019`：10进制（9不可能出现，vim自动处理）

-	`Float`

	-	`:echo 5.1e-3`：科学计数法）
	-	`:echo 5.0e3`：科学计数法中一定要有小数点

>	类型转换：Number和Float运算时会强制转换为Float

####	字符串

#####	类型转换

-	`+`、`if`这些“运算”中，vim会强制转换变量类型
	-	数字开头的字符串会转为相应`Number`（即使符合Float也
		会舍弃小数点后）
	-	而非数字开头 则转换为0

-	连接`.`
	-	`.`连接时vim可以自动将Number转换为字符串然后连接
	-	但是对于 Float，vim不能自动转换

-	转义`\`：
	-	注意`echom "foo\nbar"`类似的输出时，`echom`不会像
		`echo`一样输出两行，而是将换行输出为vim**默认**
		（即使设置了`listchars`）的“换行符”

-	字符串字面量`''`
	-	所见即所得（py中r'')，注意连续两个单引号表示单引号

-	内建字符串函数
	-	`strlen(str)`（len(str)效果对字符串同）
	-	`split(str, token=" ")`
	-	`join([str], token=" ")`
	-	`tolower(str)`
	-	`toupper(str)`

-	字符串比较`==`、`==？`、`==#`

	-	`==`：对字符串比较是否大小写敏感取决于设置
		```vimscript
		set noignorecase
		if "foo"=="Foo"
			echo "bar"（不会输出）
		endif

		set ignorecase
		if "foo"=="Foo"
			echo "bar"（会输出）
		endif
		```

	-	`==?`：对字符串比较大小写永远不敏感

	-	`==#`：对字符串比较大小写永远敏感

-	`<`、`>`：同上，也有3种

###	集合类型变量

####	列表

vim列表特点

-	有序、异质
-	索引从0开始，可以使用负数索引，使用下标得到对应元素
-	支持切割
	-	是闭区间（这个和`python`不同）
	-	可以负数区间切割
	-	可以忽略起始/结尾索引表示从0开始/末尾截至
	-	切割区间越界是安全的
	>	字符串可以像列表一样切割、索引，但是不可以使用负数
		索引，却可以使用负数切割
-	`+`用于连接两个列表

列表内建函数

-	`add(list, item)`：添加新元素
-	`len(list)`：列表长度
-	`get(list, index, default_val)`：获取列表元素，越界则
	返回`default_val`
-	`index(list, item)`：返回元素索引，不存在返回`-1`
-	`join(list, token)`：将列表中元素转换为字符串后，使用
	`toke`连接，缺省为`<space>`
-	`reverse(list)`：反转列表

####	字典

字典特性

-	值是异质的，键可以不是字符串，但是会被强制转换为字符串，
	因此，在查找值时也可以使用非字符串`dict[100]`，同样会被
	强制转换为字符串`dict["100"]`之后查找

-	支持属性`.`查找，甚至可以后接`Number`

-	添加新元素就和普通赋值一样：`let dict.100 = 100`

-	移除字典中的元素
	-	`remove(dict, index)`
	-	`unlet dict.index`/`unlet dict[index]`
	移除不存在的元素事报错

>	允许定义时多一个`,`

#####	内建函数

-	`get(dict, index, default_val)`：同列表

-	`has_key(dict, index)`：检查字典中是否有给定键，返回
	`1`（真）或`0`（假）

-	`item(dict)`：返回字典键值对，和字典一样无序

-	`keys(dict)`：返回字典所有键

-	`values(dict)`：返回字典所有值


###	作为变量的选项

-	bool选项输出0、1
	```vimscript
	:set wrap
	:set nowrap
	```

-	键值选项
	```vimscript
	:set textwidth=80
	:echo &textwidth
	```

-	本地选项（`l:`作用域下）
	```vimscript
	let &l:number=1
	```

-	选项变量还可以参与运算
	```vimscript
	let &textwidth=100
	let &textwidht = &textwidth + 10
	```

###	作为变量的寄存器

```vimscript
let @a = "hello"
echo @a
echo @"
echo @/
```

###	变量作用域

以`<char>:`开头表示作用域变量

-	变量默认为全局变量
-	`b:`：当前缓冲区作用域变量
-	`g:`：全局变量

##	语句

###	条件语句

vim中没有`not`关键字，可以使用`!`表示否定

-	`!`：否
-	`||`：或
-	`&&`：与

```vim
if "1one"
	echo "one"（会输出）
endif
if ! "one"
	echo "one"（会输出）
else
	echo "two"
endif
```

####	`finish`关键字

`finally`时结束整个vimscripts的运行

###	循环语句

####	`for`语句

```vim
let c = 0
for i in [1,2,3]
	let c+=i
endfor
echom c
```

####	`while`语句

```vim
let c = 1
let total = 0
while c<=4
	let total+=c
	let c+=1
endwhile
echom total
```

##	函数

没有作用域限制的vimscripts函数必须以大写字母开头
（有作用域限制最好也是大写字母开头）

```vim
func Func(arg1,...)
	echo "Func"
	echo a:arg1（arg1）
	echo a:0（额外（可变）参数数量）
	echo a:1（第一个额外参数）
	echo a:000（所有额外参数的list）
	return "Func"
endfunction
```

>	当`function`后没有紧跟`!`时，函数已经被定义，将会给出
	错误，而`function!`时会直接将原函数替换，除非原函数正在
	执行，此时仍然报错

-	调用方式

	-	`:call Func()`：call直接调用（return值会被直接丢弃）
	-	`:echo Func()`：表达式中调用

-	函数结束时没有return，隐式返回0

-	函数参数：最多20个

	-	参数全部位于`a:`参数作用域下
		-	`a:arg1`：一般参数
		-	`a:0`：额外（可变）参数数量
		-	`a:n`：第n个额外参数
		-	`a:000`：包含额外参数list

	-	参数不能重新赋值

vim中函数可以赋值给变量，同样的此时变量需要大写字母开头，
当然可以作为集合变量的元素，甚至也可以作为参数传递

```
function! Reversed(l)
	let nl = deepcopy(a:l)
	call reverse(nl)
	return nl
endfunction

let Myfunc = function("Reversed")

function! Mapped(func, l)
	let nl = deepcopy(a:l)
	call map(nl, string(a:func) . '(v:val)')
	return nl
endfunction

call Mapped(function("Reversed"), [[3,2], [1,2]])

let funcs = [function("Reversed"), function("Mapped")]
```

更多函数参见`functions.vim`（如果完成了）

##	Execute、Normal

-	`:execute`：把字符串当作vimscript命令执行（命令行输入）
	`:execute "echom 'hello, world'"<cr>`

	大多数语言中应该避免使用"eval"之类构造可执行字符串，但是
	vimscripts代码大部分只接受用户输入，安全不是严重问题，
	使用`:execute`命令能够极大程度上简化命令

	>	`:execute`命令用于配置文件时，不要忽略结尾的`<cr>`，
		表示“执行命令”

-	`:normal`：接受一串**键值**，并当作是normal模式接受按键

	-	`:normal`后的按键会执行映射，`:normal!`忽略所有映射

	-	`:normal`无法识别“<cr>”这样的特殊字符序列
		`:normal /foo<cr>`
		这样的命令并不会搜索，因为“没有按回车”

-	`:execute`和:normal结合使用，让`:normal`接受**按下**
	“无法打印”字符（`<cr>`、`<esc>`等），`execute`能够接受
	按键

	```vim
	:execute "normal! gg/foo\<cr>dd"
	:execute "normal! mqA;\<esc>`q"
	```

##	正则表达式

vim有四种不同的解析正则表达式的“模式”

-	默认模式下`\+`表示“一个或多个之前字符”的正常`+`意义，
	其他的符号如`{,},*`也都需要添加转义斜杠表示正常意义，
	否则表示字符字面意

	-	`:execute "normal! gg/for .\\+ in .\\+:\<cr>`：
		execute接受字符串，将`\\`转义，然后正则表达式解析，
		查找`python`的`for`语句

	-	`:execute "normal! gg".'/for .\+ in .\+:'."\<cr>"`：
		使用字符串字面量避免`\\`，但是注意此时`\<cr>`也不会
		被转义为**按下回车**（`\n`才是换行符），所以需要分开
		书写、连接

-	`\v`模式下，vim使用“very magic”**正常**正则解析模式
	-	`:execute "normal! gg".'/\vfor .+ in .+:'."\<cr>"`

