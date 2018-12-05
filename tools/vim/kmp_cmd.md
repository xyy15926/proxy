# Vim生存笔记

##	Vim模式

###	Normal模式

###	Insert模式

###	Visual模式

###	Quickfix模式

quickfix模式主要思想时保存一个位置列表，然后提供一系列命令，
实现在这个位置列表中的跳转

-	位置列表来源

	-	编译器输出信息
	-	`grep`命令输出信息（`cscope`命令）
	-	`:vimgrep`命令

-	quickfix中常用的命令有

	-	`:copen`：打开quickfix模式窗口
	-	`:cc`：显示详细错误信息
	-	`:cp`：跳至下一个错误
	-	`:cn`：跳至上一个错误
	-	`:cl`：列出所有错误
	-	`:cw`：如果有错误列表，则打开quickfix窗口
	-	`:colder/col`：到前一个旧错误列表
	-	`:cnewer/cnew`：到后一个新错误列表

###	Ex模式

###	Paste模式

##	 Motion快捷键

normal模式下移动，visual模式下选取，无特殊说明visual和normal
模式通用 

###	固定移动

####	基本

-	`hjkl`：左、上、下、右
-	`M`：窗口中间行首各非空白字符
-	`H`/`L`：窗口顶/底scrolloff处行首各非空白字符
-	`<c-i>`/`<c-o>`：跳至下一个/上一个编辑处
	（可以跨文件）

####	行内

-	`w`：下个单词开头
-	`b`/`e`：当前单词首/尾
-	`W`/`B`：下/上个非空格开头
-	`E`: 下个非空格结尾
-	`g_`：行尾，visual模式下不包括换行符
-	`$`：行尾，visual模式下包括换行符
-	`0`：行首
-	`^`：非空白字符行首
-	`|`：当前行第一列
-	`<int>|`：当前行第n列（不是字符）

####	全文

-	`gg`：文件首
-	`G`：文件尾
-	`<int>G`：第`<int>`行
-	`'<`：之前visual模式选取起始行（一直保存直至下次选取）
-	`'>`：之前visual模式选取结束行
-	`{`/`}`：下/上一个空行

###	组合

这些快捷键的效果取决于当前状态

####	行内

-	`f`/`F`：向前/后移动到某字符（`;`、`,`同向、反向重复）
-	`t`/`T`：向前/后移动到某字符前
-	`i`：在某个区域（`(), [], {}, w`等）内（显然不能单独用
	于normal）

####	全文

-	`/`/`?`：命令行向前、向后查找（`n`、`N`同向、反向重复）
-	`*`/`#`：向前、向后查找当前单词
-	`g*`/`g#`：向前、向后查找当前字符串，包括非单独单词

##	Operation快捷键

对**区域**进行操作可以和motion连用

###	内容修改

####	插入

-	`i`：insert进入insert模式
-	`I(normal)`：行首insert进入insert模式
-	`I(visual block)`：选中区域同时插入
-	`a`：add进入insert模式
-	`A`：行尾add进入insert模式
-	`<c-e>`：insert模式下复制下行当前位置字符至当前行

####	删除

-	`d`：删除选取**区域**
-	`c`：删除（修改）选中**区域**，进入insert模式
-	`x`：删除字符
-	`s`：删除字符，进入insert模式
-	`dd`：删除当前行
-	`<c-h>/<m-h>`：insert模式backspace

####	复制、粘贴

-	`["<reg>]y`：复制选中**区域**至寄存器`<reg>`，默认
	`"<reg>`
-	`yy`：复制当前行
-	`["<reg>]p`：粘贴`"<reg>`寄存器中内容，默认`"<reg>`

####	其他

-	`J`：合并下一行，`<space>`分隔
-	`u/U(visual)`：转换为小、大写
-	`~(normal)`：当前字符大、小写切换
-	`~(visual)`：选中区域大、小写切换
-	`gu/gU(normal)`：**区域**转换为小、大写
-	`g~`：**区域**大、小写切换
-	`[num]>/<`：右/左移动**区域**num个`shiftwidth`单位
-	`=`：**区域**格式化（修改为标准缩进）
-	`<c-d>`/`<c->`：insert模式减少/增加缩进

###	功能性

####	记录

-	`q<reg>`：宏记录，记录行为于寄存器`<reg>`中，按下`q`则
	停止记录
-	`@<reg>`：调用寄存器`<reg>`中的宏操作（不一定是`q<reg>`
	记录的结果）
-	`m<char>`：记录当前位置于`<char>`中（不是寄存器中未知）
-	<code>\`\<char\></code>：回到`<char>`中记录的位置

####	撤销、redo

-	`.`：重复上一个操作
-	`u`：撤销上一个操作
-	`<c-r>`：继续执行，撤销`u`
-	`<c-m>`：等同于`<cr>`

####	外部功能

-	`K`：对当前单词调用`keywordprg`设置的外部程序，默认“man”

###	界面

####	Window

-	`<c-w-h/j/k/l>`：切换至左/下/上/右窗口
-	`<c-w-w>`：轮换窗口
-	`<c-w-+/->`：竖直方向扩展/收缩当前窗口
-	`<c-w->/<>`：水平方向扩展/收缩当前窗口
-	`<c-w-=>`：恢复当前窗口高度

####	Tab

-	`[n]gt`/`gT`：下/上一个tab；第n个tab

##	Vim Cmd常用命令

###	内容相关

####	查找

#####	`/`/`?`

`:/foo`向下查找，`:?foo`向上查找

-	`set ingorecase`时，不敏感查找；`set smartcase`时，如果
	查找中有大写字符则敏感；
-	`:/foo\c`手动强制大小写不敏感，`:/foo\C`强制敏感
-	`n`、`N`重复同向、反向之前查找

#####	`:vimgrep`

文件间搜索

```md
:vim[grep] /pattern/[g][j] files
```

-	选项
	-	`g`：全局匹配（匹配每行全部）
	-	`j`：查找完毕后，进更新quickfix列表，光标不跳转

-	`files`
	-	`%`：所有缓冲区文件
	-	`**/xxxx`：当前目录及子目录所有满足`xxxx`模式文件
	-	`pattern`：满足`pattern`的文件

####	替换

#####	`:s`

```vimscripts
:[start,end]s/foo/bar[/i][I][g]
```

-	替换区域：默认只替换当前行
	-	手动指定具体行号：`:2,4s/foo/bar`（左闭右开）
	-	特殊符号
		-	全文替换`%`：`:%s/foo/bar`
		-	第一行`^`：`:^,4s/foo/bar`
		-	当前行`.`：`:.,6s/foo/bar`
		-	末尾行`$`：`:4,$s/foo/bar`
	-	visual模式下，手动选择区域命令行自动补全为`:'<,'>`
-	替换标志
	-	大小写不敏感`/i`、`\c`：`:%s/foo/bar/i`或`:%s/foo\c/bar`
	-	大小写敏感`/I`、`\C`：`%s/foo/bar/I`或`:%s/foo\C/bar`
	-	全局替换`/g`（替换每行全部模式）：`:%s/foo/bar/g`
		-	默认情况下只替换首个模式：`foo,foobar`被替换为
			`bar,foobar`
		-	全局替换模式下：`foo,foobar`被替换为`bar,barbar`
-	交互（确认）替换：`:%s/foo/bar/c`，每次替换前会询问

####	文本移动

#####	`>`/`<`

```md
:[range]>
:[range]<
```

-	范围内文本块右、左移动1个`shiftwidth`
-	移动num个`shiftwidth`添加num个`>/<`

###	文件相关

####	放弃修改、重新加载、保存

-	`:e!`：放弃本次修改（和:q!不同在于不会退出vim）
-	`:bufdo e!`：放弃vim所有已打开文件修改
-	`:e`：重新加载文件
-	`:bufdo e`：重新加载vim所有已打开文件
-	`:saveas new_file_name`：另存为，不删除原文件
-	`:sb[n]`：split窗口加载第n个buffer
-	`:b[n]`：当前窗口加载第n个buffer

####	Window

-	`:sp`：水平分割当前窗口
-	`:vs`：竖直分割当前窗口
-	`:[n]winc >/<`：水平方侧扩展/收缩
-	`:[n]winc +/-`：竖直方向扩展/收缩
-	`:res+/-[n]`：竖直方向扩展/收缩
-	`:vertical res+/-[n]`：水平方向扩展/收缩
-	`:res[n]`：高度设置为n单位
-	`:vertical res[n]`：宽度设置为n单位
-	`:q`：退出当前窗口
-	`:qa`：退出所有窗口
-	`:wq`：保存、退出当前窗口

注：

-	水平方向：扩张优先左侧变化，收缩右侧变化
-	竖直方向：扩展收缩均优先下侧变化

####	Tab

-	`:tabnew [opt] [cmd] [file_name]`：打开新tab
-	`:tabc`：关闭当前tab
-	`:tabo`：关闭其他tab
-	`:tabs`：vim cmd区域查看所有打开的tab信息
-	`:tabp`：查看前一个tab
-	`:tabfirst`/`:tablast`：第一个/最后一个tab
-	`:tabn [num]`：查看下一个/第num个tab

###	其他

####	多行语句

"|"管道符可以用于隔开多个命令`:echom "bar" | echom "foo"`

##	Vim不常用命令

-	`vim -r [file_name]`：常看交换文件（特定文件对应交换文件）
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
-	`<c-z>`暂时后台挂起vim回到shell环境中，`$fg`即可
	回到之前挂起的进程（此时为vim，详见fg命令）

###	#todo

###	#todo2


