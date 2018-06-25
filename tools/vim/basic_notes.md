# vim生存笔记

-------

##	 Motion

normal模式下移动，visual模式下选取，无特殊说明visual和normal
模式通用 

###	固定移动

-	`hjkl`：左、上、下、右
-	`w`：下个单词开头
-	`W/B`：下/上个非空格开头
-	`b/e`：当前单词首/尾
-	`g_/E`：行尾，visual模式下不包括换行符
-	`$`：行尾，visual模式下包括换行符
-	`0`：行首
-	`^`：非空白字符行首
-	`gg`：文件首
-	`G`：文件尾
-	`<int>G`：第`<int>`行
-	`|`：当前行第一列
-	`<int>|`：当前行第n列（不是字符）
-	`'<`：之前visual模式选取起始行（一直保存直至下次选取）
-	`'>`：之前visual模式选取结束行
-	`M`：窗口中间行首各非空白字符
-	`H/L`：窗口顶/底scrolloff处行首各非空白字符
-	`<ctrl-i>/<ctrl-o>`：跳至下一个/上一个编辑处
	（可以跨文件）

###	组合

-	`f/F`：向前/后移动到某字符（`;`、`,`同向、反向重复）
-	`t/T`：向前/后移动到某字符前
-	`i`：在某个区域（`(), [], {}, w`等）内（显然不能单独用
	于normal）
-	`//?`：命令行向前、向后查找（`n`、`N`同向、反向重复）
-	`*/#`：向前、向后查找当前单词
-	`g*/g#`：向前、向后查找当前字符串，包括非单独单词

##	Operation

对**区域**进行操作可以和motion连用

###	插入

-	`i`：insert进入insert模式
-	`I(normal)`：行首insert进入insert模式
-	`I(visual block)`：选中区域同时插入
-	`a`：add进入insert模式
-	`A`：行尾add进入insert模式

###	删除

-	`d`：删除选取**区域**
-	`c`：删除（修改）选中**区域**，进入insert模式
-	`x`：删除字符
-	`s`：删除字符，进入insert模式
-	`dd`：删除当前行

###	复制、粘贴

-	`["<reg>]y`：复制选中**区域**至寄存器`<reg>`，默认
	`"<reg>`
-	`yy`：复制当前行
-	`["<reg>]p`：粘贴`"<reg>`寄存器中内容，默认`"<reg>`

###	其他内容修改

-	`J`：合并下一行，`<space>`分隔
-	`u/U(visual)`：转换为小、大写
-	`~(normal)`：当前字符大、小写切换
-	`~(visual)`：选中区域大、小写切换
-	`gu/gU(normal)`：**区域**转换为小、大写
-	`g~`：**区域**大、小写切换

###	记录

-	`q<reg>`：宏记录，记录行为于寄存器`<reg>`中，按下`q`则
	停止记录
-	`@<reg>`：调用寄存器`<reg>`中的宏操作（不一定是`q<reg>`
	记录的结果）
-	`m<char>`：记录当前位置于`<char>`中（不是寄存器中未知）
-	<code>\`\<char\></code>：回到`<char>`中记录的位置

###	功能性

-	`.`：重复上一个操作
-	`u`：撤销上一个操作
-	`<c-r>`：继续执行，撤销`u`
-	`<c-m>`：等同于`<cr>`

###	其他

-	`K`：对当前单词调用`keywordprg`设置的外部程序，默认“man”

##	不常用Operation

###	界面

-	`<c-w-j>`

##	命令行常用命令

###	查找

`:/foo`向下查找，`:?foo`向上查找

-	`set ingorecase`时，不敏感查找；`set smartcase`时，如果
	查找中有大写字符则敏感；
-	`:/foo\c`手动强制大小写不敏感，`:/foo\C`强制敏感
-	`n`、`N`重复同向、反向之前查找

###	替换

`:s/foo/bar`

-	替换区域，默认只替换当前行
	-	手动指定具体行号：”:2,4s/foo/bar“（左闭右开）
	-	特殊符号
		-	全文替换`%`：`:%s/foo/bar`
		-	第一行`^`：`:^,4s/foo/bar`
		-	当前行`.`：`:.,6s/foo/bar`
		-	末尾行`$`：`:4,$s/foo/bar`
	-	visual模式下，手动选择区域命令行自动补全为`:`<,`>`
-	替换标志
	-	大小写不敏感`/i`、`\c`：`:%s/foo/bar/i`或`:%s/foo\c/bar`
	-	大小写敏感`/I`、`\C`：`%s/foo/bar/I`或`:%s/foo\C/bar`
	-	全局替换`/g`（替换每行全部模式）：`:%s/foo/bar/g`
		-	默认情况下只替换首个模式：`foo,foobar`被替换为`bar,foobar`
		-	全局替换模式下：`foo,foobar`被替换为`bar,barbar`
-	交互（确认）替换：`:%s/foo/bar/c`，每次替换前会询问

###	放弃修改、重新加载、保存

-	`:e!`：放弃本次修改（和:q!不同在于不会退出vim）
-	`:bufdo e!`：放弃vim所有已打开文件修改
-	`:e`：重新加载文件
-	`:buffdo e`：重新加载vim所有已打开文件
-	`:saveas new_file_name`：另存为，不删除原文件

###	多行语句

"|"管道符可以用于隔开多个命令`:echom "bar" | echom "foo"`

##	vim不常用命令

-	`vim -r`：常看交换文件
-	`vim -S session-file`：打开session的方式启动vim 
	-	需要配合vim命令行:mksession[!] session-file先创建
		session文件（记录vim当前的状态，包括buff区已加载文件）
	-	也可以直接进入vim，`:source session-file`载入session

##	其他

###	编辑过程中使用shell

-	`:!{shell command}`即可直接执行shell命令并暂时跳出vim
	-	`:r !{shell command}`可以将输出结果读取到当前编辑
	-	`:w !{sheel command}`可以将输出结果输出到vim命令行
-	`:shell`即可暂时进入到shell环境中，`$exit`即可回到vim中
	#todo
-	`<ctrl-z>`暂时后台挂起vim回到shell环境中，`$fg`即可
	回到之前挂起的进程（此时为vim，详见fg命令）

###	#todo

###	#todo2


