---
title: *Vim* 配置
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
updated: 2021-11-04 16:34:01
toc: true
mathjax: true
comments: true
description: vimscriptss基础--Setting
---

##	选项设置 `set`

-	*Vim* 选项：内部变量、开关，用于实现特殊效果
	-	布尔型：打开、关闭
	-	数值型
	-	字符串

> - <https://yianwillis.github.io/vimcdoc/doc/options.html>

###	选项查看

-	选项查看
	-	`:se[t][!]`：显示所有不同于缺省值的选项
		-	`!`：单独行上显示每个选项
	-	`:se[t][!] all`：显示除终端设置外所有选项
		-	`!`：单独行上显示每个选项
	-	`:se[t] termcap`：显示所有终端选项
	-	`:se[t] {OPTION}?`：显示 `OPTION` 值

-	`:opt[ions]`：打开窗口，阅读、设置所有选项
	-	窗口中可以 `<CR>` 访问链接，查看帮助信息

###	选项置位、复位

-	选项置位、复位
	-	`:se[t] {OPTION}`：布尔选项置位；数值、字符串选项显示值
	-	`:se[t] no{OPTION}`：布尔选项复位
	-	`:se[t] {OPTION}!`/`:set inv{OPTION}`：布尔选项反转
	-	`:se[t] {OPTION}&`：复位选项为缺省值
	-	`:se[t] {OPTION}&vi`：复位选项为 *Vi* 缺省值
	-	`:se[t] {OPTION}&vim`：复位选项为 *Vim* 缺省值
	-	`:se[t] all&`：复位所有选项为缺省值（除终端选项、少部分其他选项）

###	选项值设置

-	选项值设置
	-	`:se[t] {OPTION}={VALUE}`/`:se[t] {OPTION}:{VALUE}`：设置数值型、字符串型选项值
		-	数值选项值可用 10 进制、16 进制（`0x` 开头）、8 进制（`0` 开头）
		-	`OPTION` 与 `=` 可有空白（被忽略），但 `=` 和 `VALUE` 间不能有空白
	-	`:se[t] {OPTION}+={VALUE}`
		-	数值选项：将 `VALUE` 增加至数值选项
		-	字符串选项
			-	附加到字符串选项后
			-	若选项为逗号分隔列表，则补足必要逗号
			-	若选项为标志列表，会删除重复标志位（一次只应增加一个标志位）
	-	`:se[t] {OPTION}^={VALUE}`
		-	数值选项：将 `VALUE` 乘至数值选项
		-	字符串选项
			-	附加到字符串选项前
			-	若选项为逗号分隔列表，则补足必要逗号
	-	`:se[t] {OPTION}-={VALUE}`
		-	数值选项：将 `VALUE` 从数值选项中减去
		-	字符串选项
			-	若值存在：从字符串选项剔除
			-	若值不存在，无错误
			-	若选项为逗号分隔列表，则剔除不必要逗号
			-	若选项为标志位列表，`VALUE` 须和选项中出现顺序相同

-	注意事项
	-	`:set` 可设置多个参数，中途任何参数错误均报错，之后参数不处理
	-	可用 `|` 分隔 `:set` 命令
	-	字符串选项值包含特殊字符字面值需用 `\` 转义：空格、`|`、`\`、`"`
		-	对于 *Win32* 平台下的文件名，不删除普通字符前 `\`
	-	字符串选项值中 `$<VAR>` 可以扩展为环境变量

> - `:verbose set {OPTION}` 查看选项值上次设置的位置

###	选项类型

-	*Global* 全局选项：为整个 *Vim* 会话设置的选项
	-	整个 *Vim* 会话仅存在一份设置

-	*Local-Option* 局部选项：只适用于单个缓冲区（窗口）的选项
	-	局部选项包含两组值：全局值、局部值
		-	（缓冲区）全局值：可被继承，用于初始化从当前窗口创建的新缓冲区局部值
			-	每个缓冲区维护单独全局值副本，不影响其他缓冲区
			-	仅仅是用于衍生新缓冲区，也可视为是窗口全局值，对窗口（衍生缓冲区时）全局
		-	局部值：不可被继承，用于真正设置当前缓冲区
	-	缓冲区局部选项的设置逻辑
		-	*Vim* 开启时首个缓冲区执行 `$VIMRC`，初始化选项
		-	从已有窗口衍生（编辑）缓冲区，使用窗口局部值初始化局部选项
		-	曾经编辑过的缓冲区应用最近关闭的窗口选项（在缓冲区被删除后才丢失选项）

-	*Global-Local* 有局部值的全局选项：可为不同缓冲区、窗口设置不同局部值的全局选项
	-	每个缓冲区可能有单独版本、也可能使用全局选项值

###	局部值设置

-	局部选项、有局部值的全局选项设置命令
	-	`:setl[ocal][!] ...`：设置局部于当前缓冲区的值
		-	若选项选项无局部值，则设置全局值
		-	`all`：显示所有局部选项的局部值
		-	无参数：显示所有不同于缺省的局部选项的局部值，使用全局值的选项前 `--` 标记
	-	`:setl[ocal] {OPTION}<`：通过复制全局值，将局部值设置为全局值
	-	`:se[t] {OPTION}<`：（对 *Global-Local*）撤销 `OPTION` 局部值，以使用全局值
		-	对布尔型、数值型选项 `:set {OPTION}<`、`:setlocal {OPTION}<` 均为切换为全局值
		-	字符串选项还可以通过置空 `:setlocal {OPTION}=` 切换为全局值
	-	`:setg[lobal][!] ...`：只设置全局值，不改变局部值
		-	显示选项时，显示全局值

> - 可以理解为 `set` 选项必然被继承，`setlocal` 选项尽量不被继承（保存）


###	一些常用关键字

-	`iskeyword=@,_,48-57,192_255`：指定关键字
	-	下划线
	-	*ASCII* 码位在 48-57 之间的字符（`0-9`）、192-255之间的字符

-	`conceallevel=0`：隐藏等级
	-	`1`
	-	`2`

##	外观配置

-	相关选项
	-	`ruler`：右下角显示光标位置
	-	`number`：显示行号

###	 *StatusLine* 状态栏选项

-	状态栏相关选项
	-	`statusline`：设置状态栏内容
	-	`laststatus`：设置状态栏显示模式
		-	`0`：不显示
		-	`1`：默认值，两个以上窗口才显示
		-	`2`：一直显示

-	`statusline` 选项值可以包含 `printf` 风格的 `%` 项目：`%{-}{MIN_WIDTH}.{MAX_WIDTH}{ITEM}`
	-	项目格式代码
		-	`-`：左对齐（设置 `MIN_WIDTH` 时有效）
		-	`MIN_WIDTH`、`MAX_WIDHT`：最小字符数、最大字符数
		-	数字内容可在 `%` 后跟 `0` 以填充前导零
		-	`%%`：转义百分号
		-	`\ `：空格需用 `\` 转义
	-	控制代码
		-	`%=`：紧贴左、紧贴右分割点
		-	`%{N}*`：设置高亮级别（直到下个高亮设置）
		-	`%<`：状态行过长时换行位置
		-	`%{EXPR}`：表达式 `EXPR` 结果
			-	`%{&ff}`：文件格式（`DOS`、`UNIX`）
	-	文件信息、编辑状态
		-	`%n`：缓冲区编号
		-	`%f`、`%t`、`%F`：缓存区文件路径（`:ls` 路径）、文件名、完整路径文件名
		-	`%y`、`%Y`：缓冲区文件类型：`[python]`、`PYTHON` 形式
		-	`%b`、`%B`：当前字符 10 进制值、16 进制值
		-	`%m`：缓冲区（相对已保存文件）修改标记：`[+]` 已修改、`[-]` 不可修改
		-	`%l`、`%L：当前行号、缓冲区行数
		-	`%v`、`%c`、`%V`：虚列号、列号、列数
		-	`%p`：行位置百分比
		-	`%O`：16 进制表示的当前字符偏移位置
		-	`%N`：打印机页号
		-	`%R`、`%r`：缓冲区只读则 `RO`、`[ro]`
		-	`%W`、`%w`：为预览窗口则 `PRV`、`[Preview]`
		-	`%H`、`%h`：帮助缓冲区则 `HLP`、`[Help]`

	```vimscripts
	set statusline=%f\ -\ filetype:\ %y
	set statusline=%f
	set statusline+=%=
	set statusline+=%l
	set statusline+=/
	set statusline+=%L
	```

###	*Wildkey* 自动补全

-	配置选项
	-	`wildmenu`：启用增强模式的命令行补全
	-	`wildmode`：补全模式
		-	`""`：仅使用首个匹配结果
		-	`full`：遍历匹配
		-	`longest`：使用最长的公共子串补全
		-	`list`：在 *wildmenu* 中显示匹配的文件列表
		-	可选项合并使用
			-	`longest,full`：显示候选，并用最长公共子串补全
			-	`list,full`：显示候选，并使用首个匹配项补全
			-	`list,longest`：显示候选，并使用最长子串补全
			-	`list,longest,full`：显示匹配的文件列表，并使用最长子串补全，之后遍历匹配
	-	`wildignore`：候选项中将忽略的名称模式
		-	可使用通配符
		-	`,` 分隔的列表
	-	`wildchar`：自动补全触发键
		-	*Vim* 模式缺省为 `<Tab>`，*Vi* 模式缺省为 `<Ctrl-E>`

> - `suffixes` 选项中包含的后缀类型文件在候选项中优先级较低
> - <https://zhuanlan.zhihu.com/p/87021392>
