---
title: *Vim* 编辑 - 插入、缓冲区
categories:
  - Linux
  - Tool
  - Vi
tags:
  - Linux
  - Tool
  - Vi
  - Editing
date: 2021-11-08 11:33:47
updated: 2021-12-02 11:06:51
toc: true
mathjax: true
description: 
---

##	缓冲区（文件）编辑

-	*Vim* 编辑文件过程
	-	将文件读取至缓冲区
	-	用编辑器命令修改缓冲区
	-	将缓冲区内容写回文件：保存缓冲区前，文件内容不改变

###	缓冲区信息

-	缓冲区名称
	-	缓冲区列表：记录所有缓冲区名称，可用 `:ls` 打印
		-	编辑、写回时，对应文件名被加入缓冲区列表，作为缓冲区名称
	-	当前文件名：当前在编辑缓冲区名称
		-	在 `:ls` 结果中用 `%` 标识，可用 `%` 引用
		-	保存缓冲区时，缺省使用当前文件名
	-	轮换文件名：用于轮换的缓冲区名称
		-	在 `:ls` 结果中用 `#` 标识，可用 `#` 引用
	-	每个窗口维护独立的当前文件名、轮换文件名

-	相关命令
	-	`:ls`、`:buffers`、`:files`：列出缓冲区列表
	-	`:keepalt {CMD}`：执行 `CMD`，在此期间保持轮换文件名不变
	-	`:[0]f[ile][!] [NAME]`：显示当前文件名、光标位置、文件状态
		-	`!`：不截断消息，即使置位 `shortmess`
		-	`NAME`：设置当前缓冲区名称
		-	`0`：删除当前缓冲区名称
	-	`{COUNT}CTRL-g`：显示当前文件名、光标位置、文件状态
		-	`COUNT`：给出完整文件路径，若计数大于 1，同时给出缓冲区号
	-	`g CTRL-g`：显示当前光标位置，分别按：列、和、单词、字符、字节计数
	-	`v_g CTRL-g`：显示可视区域单词、字符、行、字节计数

-	相关选项
	-	`backup`：文件被覆盖前备份文件
	-	`backupext`：备份文件名后缀（备份文件名为原文件名 + 此后缀）
	-	`shortname`：替换多余点好为 `_`
	-	`backupdir`：备份文件存储路径
	-	`shortmess`：
	-	`autowriteall`：自动写回改动
	-	`hidden`：允许对缓冲区修改且不写回后将其切换至后台（隐藏）

####	编辑文件

-	相关命令
	-	`:e[dit|x][!] [++<OPT>] [+<CMD>] [FILE] [#<COUNT>]`：编辑文件
		-	`!`：放弃缓冲区改动
	-	`:vi[sual][!] [++<OPT>] [+<CMD>] [FILE] [#<COUNT>]`：*Ex* 模式时退出 *Ex* 模式，否则同 `edit`
	-	`:vie[w][!] [++<OPT>] [+<CMD>] [FILE] [#<COUNT>]`：*Ex* 模式时退出 *Ex* 模式，否则同只读 `edit`
	-	`[COUNT]CTRL-^`：切换缓冲区
		-	`COUNT`：缓冲区编号，缺省为轮换缓冲区（等价于 `:e #`，但可切换无名缓冲区）
	-	`:ene[w][!]`：编辑新的无名缓冲区
		-	`!`：放弃缓冲区改动
	-	`:[COUNT]fin[d][!] [++OPT] [+CMD] [FILE]`：在 `path` 中寻找 `FILE` 并编辑
		-	`COUNT`：编辑第 `COUNT` 个匹配
	-	`[COUNT]gf`：编辑光标上对应文件名的文件
		-	`COUNT`：编辑第 `COUNT` 个匹配
	-	`v_[COUNT]gf`：选中内容作为文件名（此时忽略 `isfname` 选项）
	-	`[COUNT]gF [NO]`：同 `gf`，但跳转至 `NO` 行
	-	`v_[COUNT]gF [NO]`：同 `v_[COUNT]gf`
	-	`CTRL-w CTRL-f`：新窗口编辑 `:gf`

-	通用参数说明
	-	`FILE`：编辑文件名，缺省重新载入当前文件
		-	*Wildcards* 方式搜索、匹配文件：其中通配符被扩展，具体支持取决于平台
		-	反引号后紧跟 `=` 将被视为 *Vim* 表达式被计算
		-	*Unix* 平台上，可用反引号括起 Shell 命令，将其输出作为结果
	-	`COUNT`：缓冲区编号，缺省为轮换缓冲区
	-	`OPT`：指定 `fileformat`、`fileencoding`、`binary`、坏字符处理方案
		-	`++fileformat`/`++ff`
		-	`++encoding`/`++enc`
		-	`++binary`/`++bin`、`++nobinary`/`++nobin`
		-	`++bad=<X>|keep|drop`：用 `X` 替换、维持、删除
	-	`CMD`：在新打开文件中定位光标、执行命令
		-	`+`：从最后一行开始
		-	`+{NUM}`：从第 `NUM` 行开始
		-	`+/{PTN}`：从首个匹配 `PTN` 开始
		-	`+{CMD}`：打开文件后执行 *Ex* 命令

-	相关选项
	-	`isfname`：决定组成文件名的字符
	-	`suffixesadd`：查找文件所需附加的后缀
	-	`fileformat`：换行符风格（可通过读取、写入时设置不同风格实现转换）
		-	`dos`：`<CR><NL>`、`<NL>`
		-	`unix`：`<NL>`
		-	`mac`：`<CR>`

> - *US* 键盘上 `^` 在 `6` 上，`CTRL-6` 会给出 `CTRL-^` 键码

####	参数列表

-	参数列表：启动 *Vim* 时给出的多个文件名
	-	不同于 `:buffers` 的缓冲区列表，参数列表中文件名在缓冲区列表中存在，反之不然
	-	所有窗口缺省使用相同全局参数列表，但可以通过 `:arglocal` 创建局部

-	全局参数列表相关命令
	-	`:ar[gs][!] [++<OPT>] [+<CMD>] [ARGLIST]`：显示参数列表，当前文件方括号标识
		-	`ARGLIST`：作为新的参数列表并编辑
			-	`##` 引用当前参数列表：`:args ## x`
	-	`:[COUNT]arge[dit][!] [++<OPT>] [+<CMD>] {FILE}`：将 `FILE` 加入参数列表
		-	`FILE`：可为多个文件，若已存在则直接编辑
		-	`COUNT`：加入至第 `COUNT` 位，缺省当前文件之后
			-	`$`：最后
	-	`:[COUNT]args[dd][!] [FILE]`：将 `FILE` 添加至参数列表中
		-	`FILE`：可为多个文件，不检查重复项
	-	`:[RANGE]argd[elete] [PTN]`：从参数列表中的删除文件
	-	`:[COUNT]argu[ment][!] [COUNT] [++<OPT>] [+<CMD>]`：编辑参数列表中第 `COUNT` 个文件
	-	`:[COUNT]n[ext][!] [++<OPT>] [+<CMD>] [ARGLIST]`：编辑向后第 `COUNT` 个文件
		-	`ARGLIST`：给出时同 `:args`
	-	`:[COUNT]N[ext][!] [COUNT] [++<OPT] [+<CMD>]`：编辑向前第 `COUNT` 个文件
	-	`:[COUNT]prev[ious][!] [COUNT] [++<OPT] [+<CMD>]`：编辑向前第 `COUNT` 个文件
	-	`:rew[ind][!] [++<OPT>] [+<CMD>]`：编辑参数列表中首个文件
	-	`:fir[st][!] [++<OPT>] [+<CMD>]`：同 `:rewind`
	-	`:la[st][!] [++<OPT>] [+<CMD>]`：编辑参数列表中最后文件
	-	`:[COUNT]wn[ext][!] [++<OPT>] [FILE]`：写回当前缓冲区，并开始编辑向后 `COUNT` 个文件
		-	`FILE`：将当前文件写入至 `FILE` 中
	-	`:[COUNT]wN[ext][!] [++<OPT>] [FILE]`：写回当前缓冲区，并开始编辑向前 `COUNT` 个文件
	-	`:[COUNT]wp[revious][!] [++<OPT>] [FILE]`：写回当前缓冲区，并开始编辑向前 `COUNT` 个文件

-	局部参数列表
	-	`:argl[ocal][!] [++<OPT>] [+<CMD>] [ARGLIST]`：定义局部于当前窗口的参数列表
		-	`ARGLIST`：局部参数列表，缺省复制全局参数列表
	-	`:argg[lobal][!] [++<OPT>] [++<CMD>]`：定理全局参数列表，影响所有窗口
		-	`ARGLIST`：全局参数列表，缺省直接使用全局参数列表

-	使用参数列表
	-	`:[RANGE]argdo[!] <CMD>`：对范围内参数列表执行 `CMD`
		-	说明
			-	`CMD` 不可修改参数列表
			-	命令执行是，`Syntax` 自动命令事件被加入 `eventignore` 中，提高编辑速度

-	通用参数说明
	-	`!`：忽略缓冲区更改
	-	`PTN`：删除匹配 `PTN` 的文件
		-	`%`：当前项
	-	`RANGE`：删除 `RANGE` 范围里的文件，缺省全部 `1,$`
		-	`$`：最后项
		-	`.`、空白：删除当前
		-	`%`：全部项
	-	`FILE`：编辑文件名，缺省重新载入当前文件
		-	其中通配符被扩展，具体支持取决于平台
		-	反引号后紧跟 `=` 将被视为 *Vim* 表达式被计算
		-	*Unix* 平台上，可用反引号括起 Shell 命令，将其输出作为结果
	-	`COUNT`：参数列表编号，缺省为轮换缓冲区
	-	`OPT`：指定 `fileformat`、`fileencoding`、`binary`、坏字符处理方案
		-	`++fileformat`/`++ff`
		-	`++encoding`/`++enc`
		-	`++binary`/`++bin`、`++nobinary`/`++nobin`
		-	`++bad=<X>|keep|drop`：用 `X` 替换、维持、删除
	-	`CMD`：在新打开文件中定位光标、执行命令
		-	光标定位
			-	`+`：从最后一行开始
			-	`+{NUM}`：从第 `NUM` 行开始
			-	`+/{PTN}`：从首个匹配 `PTN` 开始
			-	`+{CMD}`：打开文件后执行 *Ex* 命令
		-	执行命令
			-	可用 `|` 分隔多个命令

####	写入

-	写入命令
	-	`:[RANGE]w[rite][!] [++<OPT>] [>>] [FILE]`：将缓冲区指定内容写入文件
	-	`:[RANGE]w[rite] [++<OPT>] !{CMD}`：将缓冲区指定内容作为标准输入执行 `CMD`（Shell 命令）
	-	`:sav[as][!] [++<OPT>] <FILE>`：用 `FILE` 保存缓冲区，并作为当前缓冲区名
	-	`:[RANGE]up[date][!] [++<OPT>] [>>] [FILE]`：仅在缓冲区修改才写入
	-	`:wa[ll][!]`：保存所有已修改缓冲区
	-	`:recover`：从交换文件中恢复缓冲区


-	通用参数说明
	-	`!`：强制写入，即使置位 `readonly`、文件（不）已存在、无法创建备份文件
		-	可能会破坏文件、权限位
	-	`RANGE`：指定缓冲区范围写入，缺省整个缓冲区
	-	`>>`：将缓冲区内容附加到文件后
	-	`FILE`：写入目标文件，缺省缓冲区名称
		-	`FILE` 被给出时，与当前缓冲区名将互为轮换文件
	-	`OPT`：指定 `fileformat`、`fileencoding`、`binary`
		-	`++fileformat`/`++ff`
		-	`++encoding`/`++enc`
		-	`++binary`/`++bin`、`++nobinary`/`++nobin`

-	相关选项
	-	`write`：允许写入文件
	-	`backup`：备份原文件
	-	`writebackup`：写入时备份，写入完成后删除备份文件
	-	`backupskip`：备份时忽略匹配的文件名
	-	`backupdir`：存放备份路径，缺省为写入文件相同目录
	-	`backupcopy`：决定复制、改名实现备份

-	说明
	-	*Vim* 写入新文件时权限设置为可读写，写入已读入文件时将保留原始权限，但清除 `s` 权限
	-	若写入文件名是设备名，*Vim* 无法建立备份，同时必须使用 `!` 强制写入

####	退出、写入退出

-	退出命令
	-	`:[conf[irm]] q[uit][!]`：退出当前窗口
		-	若为最后的编辑窗口时，同时退出标签页、*Vim*
		-	缺省：退出若缓冲区不可放弃、有未编辑参数列表，操作失败
		-	`!`：退出不保存，即使缓冲区有修改
		-	`conf`：若有已修改缓冲区、未编辑参数楼列表等影响退出原因，给出提示
	-	`:cq[uit]`：任何情形下退出不保存，并返回错误代码
	-	`:[RANGE]wq[!] [++<OPT>] [FILE]`：写回文件并关闭窗口
		-	`!`：退出，即使有未编辑参数列表
	-	`:[RANGE]x[it][!] [++<OPT>] [FILE]`：同 `:wq`，但仅在文件有修改时写回
	-	`:[RANGE]exi[t][!] [++<OPT>] [FILE]`：同 `:x`
	-	`ZZ`：同 `:x`
	-	`ZQ`：同 `:q!`
	-	`:[conf[irm]] q[uit]a[ll][!]`：退出 *Vim*（可选项同 `:q`）
	-	`:[conf[irm]] wqa[ll][!] [++<OPT>]`：保存修改过的缓冲区，并退出 *Vim*
		-	`!`：写回包括只读缓冲区，但若存在无名、其他原因写回失败，退出依然失败
		-	`conf`：若有已修改缓冲区、未编辑参数楼列表等影响退出原因，给出提示
	-	`:[conf[irm]] xa[ll][!] [++<OPT>]`：同 `:wqa`
	-	`:close`：关闭当前窗口
		-	`:quit` 退出缓冲区编辑也都关闭窗口，但 `:close` 可避免关闭最后窗口
	-	`:only`：关闭其他窗口

-	通用参数解释
	-	`RANGE`：指定缓冲区范围（行）写入，缺省整个缓冲区
	-	`>>`：将缓冲区内容附加到文件后
	-	`FILE`：写入目标文件，缺省缓冲区名称
	-	`OPT`：指定 `fileformat`、`fileencoding`、`binary`
		-	`++fileformat`/`++ff`
		-	`++encoding`/`++enc`
		-	`++binary`/`++bin`、`++nobinary`/`++nobin`

###	对话框

-	对话框命令
	-	`:conf[irm] <CMD>`：执行 `CMD`，若存在待确认事项，显示对话框
		-	用于 `:q`、`:qa`、`:w` 及其他会以类似方式失败的命令，如：`:only`、`:buffer`、`bdelete`
	-	`:bro[wse] <CMD>`：为（支持浏览的） `CMD` 的参数显示选择对话框
		-	文件浏览：用于 `:e`、`:w`、`:wall`、`:mkexrc`、`:mkvimrc`、`:split`、`:cgetfile` 等命令
		-	可用 `g:browsefilter`、`b:browsefilter` 变量过滤选项：`<TAG>\t<PTN>;<PTN>\n`
			-	`TAG`：*File of Type* 组合框中文字
			-	`PTN`：过滤文件名的模式，多个模式用 `;` 分隔
			-	多个选项直接可直接合并
		-	`:browse set`：类似 `:options`

> - `:browse` 浏览文件需要 `+browse` 特征

###	当前目录

-	目录命令
	-	`:cd[!] <PATH>`：改变当前目录
	-	`:chd[ir][!] <PATH>`：同 `:cd`
	-	`:tcd[!] <PATH>`：类似 `:cd`，仅为当前标签页设置当前目录
	-	`:tch[dir][!] <PATH>`：同 `:tcd`
	-	`:lcd[!] <PATH>`：类似 `:cd`，仅为当前窗口设置当前目录
	-	`:lch[dir][!] <PATH>`：同 `:lcd`
	-	`:pw[d]`：显示当前目录名

-	参数说明
	-	`PATH`：目标目录（基本同 `$ cd` 命令）
		-	缺省 *Unix* 上改变当前目录到主目录，非 *Unix* 显示当前目录名
		-	`-`：切换到上个当前目录
		-	若为相对路径，则在 `cdpath` 列出的目录中搜索

-	目录选项
	-	`cdpath`：`:cd` 命令查找相对路径的路径

> - 对当前目录的修改不改变当前已经开文件，但是可能会改变参数列表中文件

###	编辑二进制文件

-	*Vim* 用 `-b` 表示以二进制模式进行文件读写
	-	相关的文件编辑选项被设置
		-	`binary`
		-	`textwidth=0`
		-	`nomodeline`
		-	`noexpandtab`

###	加密

-	*Vim* 支持文件加密独写
	-	但常规的选项设置存在问题
		-	交换文件、撤销文件此时被分块加密
		-	内存中文本、`:!filter`、`:w {CMD}` 过滤文本未加密
		-	*viminfo* 文本未加密
	-	加密文件头部有魔术数字，*Vim* 据此确认加密文件
	-	可将如下配置写入 *magic* 文件（`/etc/magic`、`/usr/share/misc/magic`）使得加密文件可被 `file` 命令识别

		```cnf
		0	string	VimCrypt~	Vim encrypted file
		>9	string	01			- "zip" cryptmethod
		>9	string	02			- "blowfish" cryptmethod
		>9	string	03			- "blowfish2" cryptmethod
		```

-	加密命令
	-	`:X`：提示输入加密密钥

-	相关选项
	-	`key`：存储密钥
		-	写入时若该选项非空，则用其值作为密钥加密，否则不加密
		-	读取时若该选项非空，使用其值解密，否提示输入密钥
	-	`cryptmethod`/`cm`：加密方法（须在写入文件前设置）
		-	`zip`：弱加密
		-	`blowfish`：有漏洞
		-	`blowfish2`：中强度

###	修改时间

-	*Vim* 会检查 *Vim* 之外的文件修改，避免文件的不同版本
	-	记住文件开始开始编辑时的修改时间、模式、大小
	-	执行 Shell 命令（`:!<CMD>`、`:suspend`、`:read!`、`K`）后
		-	*Vim* 比较缓冲区的修改时间、模式、大小
		-	并对修改的文件执行 `FileChangedShell` 自动命令、或显示警告
	-	若文件在缓冲区中未经编辑，*Vim* 会自动读取、比较

-	相关命令
	-	`:[N]checkt[ime] [FILE] [N]`：检查文件是否在 *Vim* 外被修改
		-	`N`、`FILE`：检查特定编号、名称缓冲区，缺省全部
		-	在自动命令、`:global` 命令、非键盘输入中调用时，实际检查会被延迟，直到副作用没有问题
		-	检查之后会采取自动读取、警告、错误

-	相关选项
	-	`autoread`：自动载入在 *Vim* 外修改的文件
		-	仅在缓冲区未被更改时生效
	-	`buftype`
		-	`nofile`：不询问文件更新警告

###	文件搜索、*Wildcards*

-	文件搜索：`path`、`cdpath`、`tags` 选项值设置，以及 `finddir()`、`findfile()` 搜索文件的逻辑
	-	向下搜索：可使用 `*`、`**` 或其他操作系统支持的通配符
		-	`*`：匹配 0 个或更多字符
		-	`**[N]`：匹配 `N` 层目录，缺省 30 层
	-	向上搜索：给定起、止目录，沿目录树向上搜索
		-	起始目录、多个终止目录 `;` 分隔
	-	混合向上、向下搜索
		-	若 `set path=**;/path`，当前目录为 `/path/to/current`，则须在 `path` 中搜索时搜索
			-	`/path/to/current/**`
			-	`/path/to/**`
			-	`/path/**`

-	*Wildcards* 匹配：其余 *Ex* 命令均使用此方式匹配文件（无需预设搜索范围）
	-	具体支持取决于平台，但以下通用
		-	`?`：一个字符
		-	`*`：任何东西，包括空
		-	`**`：任何东西，包括空，递归进入目录
		-	`[abc]`：`a`、`b` 或 `c`

##	插入、替换模式

-	替换模式 `R`：输入的每个字符会删除行内字符，直至在末尾附加输入的字符
	-	若输入 `<NL>`，则插入换行符，不删除任何字符
	-	`<Tab>` 是单个字符，但占据多个位置，可能影响窗口展示
	-	用 `<BS>`、`CTRL-w`、`CTRL-u` 删除字符，实际上是删除修改，被替换的字符将复原

-	虚拟替换模式 `gR`：类似替换模式，但按屏幕位置替换，保持窗口中字符不移动
	-	`<Tab>` 可能会替换多个字符，在 `<Tab>` 上替换可能等同于插入
	-	`<NL>` 会替换光标至行尾
	-	用 `<BS>`、`CTRL-w`、`CTRL-u` 删除字符，实际上是删除修改，被替换的字符将复原
	-	适合用于编辑 `<Tab>` 分隔表格列的场合

###	普通模式插入

-	普通模式进入插入模式命令
	-	`a`：光标后附加文本 `COUNT` 次
	-	`A`：行尾附加文本 `COUNT` 次
	-	`<insert>`/`i`：光标前插入文本 `COUNT` 次
	-	`I`：行首非空白字符前插入文本 `COUNT` 次
	-	`gI`：首列插入文本 `COUNT` 次
	-	`gi`：缓冲区最近一次插入模式停止处继续插入文本
		-	位置记录在 `^` 位置标记处
	-	`o`/`O`：光标下方、下方开启新行，插入文本 `COUNT` 次

-	*Ex* 插入命令
	-	`:[RANGE]a[ppend][!]`：指定行下方添加行，持续输入直至输入仅包含 `.` 行
		-	`!`：切换 `autoindent` 选项
	-	`:[RANGE]i[nsert][!]`：指定行上方添加行，持续输入直至输入仅包含 `.` 行
	-	`:star[tinsert][!]`：进入插入模式
		-	`!`：类似 `A`，否则同 `i`
		-	命令不能在 `:normal` 中使用
	-	`:stopi[nsert]`：尽快停止插入模式，类似 `<Esc>`
	-	`:startr[eplace][!]`：启动替换模式，类似 `R`
		-	`!`：类似 `$R`
	-	`:startg[replace][!]`：启动虚拟替换，类似 `gR`

-	插入文件命令
	-	`:[RANGE]r[ead] [++<OPT>] [!<CMD>] [NAME]`：光标下插入内容
		-	`NAME`：文件包含内容
		-	`CMD`：执行 `CMD` 的标准输出
			-	`shellredir` 选项用于保存命令的输出结果

-	通用参数列表
	-	`RANGE`：添加新行位置，缺省为当前行
	-	`OPT`：指定 `fileformat`、`fileencoding`、`binary`、坏字符处理方案
		-	`++fileformat`/`++ff`
		-	`++encoding`/`++enc`
		-	`++binary`/`++bin`、`++nobinary`/`++nobin`
		-	`++bad=<X>|keep|drop`：用 `X` 替换、维持、删除

###	插入模式插入

-	简单编辑
	-	`<Insert>`：切换插入、替换模式
	-	`<Esc>`/`CTRL-[`：回到普通模式
	-	`CTRL-c`：回到普通模式，不检查缩写、不激活 `InsertLeave` 自动命令事件
	-	`<BS>`/`CTRL-H`：删除光标前字符
	-	`<Del>`：删除光标下字符
	-	`<CTRL-w>`：删除光标前单词
	-	`<CTRL-u>`：删除当前行光标前全部字符
	-	`<Tab>`/`CTRL-i`：插入制表符
		-	可用 `CTRL-v <Tab>` 避免制表符扩展
	-	`CTRL-k <CHAR1> <CHAR2>`：输入二合字母
		-	`:digraphs` 查看二合字母
	-	`CTRL-t`/`CTRL-i`：当前行开始处插入 `shiftwidth` 的缩进
	-	`CTRL-d`：当前行开始处删除 `shiftwidth` 的缩进
		-	`0 CTRL-d`：当前行开始处删除全部 `shiftwidth` 的缩进
		-	`^ CTRL-d`：当前行开始处删除全部 `shiftwidth` 的缩进，下行恢复
	-	`CTRL-e`、`CTRL-y`：插入光标下方、上方字符
		-	下方、上方指页面行

-	特殊插入
	-	`CTRL-@`：插入最近插入文本，并停止插入
	-	`CTRL-a`：插入最近插入文本
	-	`CTRL-r <REG>`：类似键盘输入插入寄存器 `REG` 内容
	-	`CTRL-r CTRL-r <REG>`：按本义插入寄存器 `REG` 内容
	-	`CTRL-r CTRL-0 <REG>`：按本义插入寄存器 `REG` 内容，并且不进行自动缩进
		-	同鼠标粘贴文本 `<MiddleMouse>`
		-	若寄存器面向行，在当前行上插入文本，类似 `P`
		-	`.` 寄存器仍然类似键盘输入方式插入
	-	`CTRL-r CTRL-p <REG>`：按本义插入寄存器 `REG` 内容，修正缩进
		-	同鼠标粘贴文本 `<MiddleMouse>`
		-	若寄存器面向行，在当前行上插入文本，类似 `P`
		-	`.` 寄存器仍然类似键盘输入方式插入

-	其他输入
	-	`CTRL-_`：切换语言（及输入方向）
		-	需 `allowrevins` 置位、编译时加入 `+rightleft` 特性
	-	`CTRL-^`：切换字符输入使用方式？？？？？
	-	`CTRL-]`：触发缩写，不插入字符

-	相关选项
	-	`modifyOtherKeys`：插入带修饰符的键的转义序列
	-	`backspace`：指定退格功能，逗号分隔的项目
		-	`indent`：允许退格删除自动缩进
		-	`eol`：允许退格删除换行符
		-	`start`：允许退格删除（本次）插入前的位置；`CTRL-w`、`CTRL-u` 在开始位置停止
	-	`textwidth`：断行长度
		-	输入边界后的非空白字符后断开
		-	仅在插入模式、行后附加才会自动断开
	-	`formatoptions`：限制断行时机
		-	`l`：插入开始时，文本行长度不超过 `textwidth` 时，断行才发生
		-	`v`：只在当前插入中输入空白字符上断行
		-	`lv`：插入开始时文本长度不超过 `textwidth`、且在当前插入中输入空白字符才断行
	-	`formatexpr`：断行处理函数
	-	`wrapmargin`：根据屏幕宽度动态断行，即等价于 *columns* - `textwidth`
	-	`linebreak`：文本回绕


###	特殊字符

-	相关命令
	-	`CTRL-v`：插入特殊字符，非数字则按本义插入
		-	特殊键：插入终端代码
		-	数字：应为字符的 10、8、16 进制值
			-	`XXXX`：10 进制，最大值 `255`
			-	`oXXXX`、`OXXXX`：8 进制，最大值 `777`
			-	`xXXXX`、`XXXXX`：16 进制，最大值 `ff`
			-	`uXXXX`：16 进制，最大值 `ffff`
			-	`UXXXX`：16 进制，最大值 `7fffffff`
	-	`CTRL-q`：同 `CTRL-v`，但可能被终端吃掉
	-	`CTRL-SHIFT-v`/`CTRL-SHIFT-q`：类似 `CTRL-v`	
		-	除非激活 `modifyOtherKeys`，此时插入带修饰符的键的转义序列

####	回车、换行

-	回车、换行说明
	-	三种平台文本格式的 *EOL* 标识
		-	`<NL><CR>`：*MS-DOS*
		-	`<NL>`：*Unix*
		-	`<CR>`：*Macintosh* （*OSX* 前系统）
	-	*Vim* 对 `<NL>`、`<CR>` 处理
		-	文本格式下用于换行的控制字符被转换为 `<Nul>`（*ASCII* 码值 0）插入缓冲区
			-	`ff=mac` 无法插入 `<CR>`/`^M`
			-	`ff=unix`、`ff=dos`  无法插入 `<NL>`/`^J`
		-	`unix`、`dos` 与 `mac` 格式互转时，文本结构不发生变化
			-	可认为存在 `<EOL>` 标记，在缓冲区切换格式时，在 `<NL><CR>`、`<NL>`、`<CR>` 之间转换
			-	多余的、显示出的 `<CR>` 与 `<NL>` 互相转化
			-	可利用此特性在文本中插入 `<NL>`、`<CR>`

-	相关命令
	-	`i_CTRL-j`/`<NL>`：开始新行
	-	`i_CTRL-m`/`<CR>`：开始新行

-	相关选项
	-	`fileformats`：尝试用于打开文件的文本格式
		-	`unix`、`dos`、`mac`
	-	`fileformat`：当前缓冲区的文本格式

> - `<NL>` *New Line* 即 `<LF>` *Line Feed*，即 *ASCII* 码值 14 的字符

####	缩进、制表符

-	相关命令
	-	`:[RANGE]ret[ab][!] [NEW_TABSTOP]`：将空白序列替换为 `NEW_TABSTOP` 确定的空白序列
		-	根据 `expandtab` 选项决定替换为 `<Tab>`、空格
		-	`NEW_TABSTOP`：缺省、0 则使用 `tabstop` 选项值
		-	已有 `<Tab>` 用 当前 `tabstop` 选项值确定
		-	`!`：把包含正常空格字符串替换为 `<Tab>`

-	相关选项
	-	`tabstop`：制表符真实占位（影响换算、展示）
	-	`softtabstop`：（非 0 时）`<Tab>` 插入、`<BS>` 删除的位置数量
	-	`expandtab`：空格填充制表符位置
		-	置位时可用通过 `CTRL-v <Tab>` 输入真实 `Tab`
	-	`smarttab`：`<Tab>` 在行首插入 `shiftwidth` 个位置，在其他地方插入 `tabstop` 个位置
	-	`shiftwidth`：缩进位置数量，影响 `>>` 等命令

###	动作-插入字符

-	以下字符行为类似停止插入模式、动作、继续插入
	-	`<Up>`、`<Down>`：光标上移、下移
	-	`CTRL-g <Up>`/`CTRL-g k`/`CTRL-g CTRL-k`、`CTRL-g <Down>`/`CTRL-g j`/`CTRL-g <Up>`：光标上移、下移，首行插入
	-	`<Left>`、`<Right>`：光标左移、右移
	-	`<S-Left>`/`<C-Left>`、`<S-Right>`/`<C-Right>`：光标反向、正向移动一个单词
	-	`<HOME>`：光标移至行首
	-	`<END>`：光标移至行末
	-	`<C-HOME>`：光标移至文件首
	-	`<C-END>`：光标移至文件末
	-	`<S-Up>`/`<PageUp>`、`<S-Down>`/`<PageDown>`：上翻、下翻窗口一页
	-	`CTRL-o`：执行命令，然后回到插入模式
		-	副作用：若光标在行尾外，会先移动到行最后字符上
	-	`CTRL-\ CTRL-o`：类似 `CTRL-o`，但不移动光标
	-	`CTRL-l`：置位 `insertmode` 时，转到普通模式
	-	`CTRL-G u`：打断撤销序列，开始新的改变
	-	`CTRL-G U`：光标停在同一行，下个左、右光标移动不打断撤销

-	鼠标
	-	`<LeftMouse>`：光标移至鼠标点击处
	-	`<ScrollWheelDown>`、`<ScrollWheelUp>`、`<ScrollWheelLeft>`、`<ScrollWheelRight>`：窗口向下、上、左、右滚动三行
	-	`<S-ScrollWheelDown>`、`<S-ScrollWheelUp>`、`<S-ScrollWheelLeft>`、`<S-ScrollWheelRight>`：窗口向下、上、左、右滚动整页

###	插入补全

-	`CTRL-x CTRL-e`、`CTRL-x CTRL-y`：窗口向上、向下滚动一行
	-	`CTRL-n`：查找下个关键字
	-	`CTRL-p`：查找下个关键字

-	`CTRL-x` 可以进入插入补全子模式：可给出命令补全单词、滚动窗口
	-	若补全处于激活状态，`CTRL-e` 停止补全并回到原录入文字
	-	若弹出菜单出现，`CTRL-y` 停止补全并接受当前选择项
	-	输入 `CTRL-x`、`CTRL-n`、`CTRL-p`、`CTRL-r` 外按键将退出补全模式
		-	`CTRL-x`：进入补全模式
		-	`CTRL-n`：下个候选项，从 `complete` 选项给出位置搜索下个匹配
		-	`CTRL-p`：上个候选项，从 `complete` 选项给出位置搜索下个匹配
		-	`CTRL-r`：插入寄存器内容（主要是为允许通过 `=` 寄存器调用函数决定下个操作）
	-	输入非退出按键不被映射
		-	如：`:map ^F ^X^F` 可以正常工作（`^F` 不会被继续映射）
	-	补全命令后
		-	再次输入补全命令 `CTRL-<X>` 将匹配下个候选项，同 `CTRL-n`
		-	输入补全命令 `CTRL-x CTRL-<X>` 将尝试将复制匹配项的下文，直至输入两次 `CTRL-x`

-	插入补全子模式补全类型
	-	`CTRL-x CTRL-l`：反向搜索补全整行，忽略缩进
		-	`complete` 决定匹配搜索的缓冲区
	-	`CTRL-x CTRL-n`、`CTRL-x CTRL-p`：正向、反向搜索补全单词
	-	`CTRL-x CTRL-k`：根据 `dictionary` 选项补全单词
	-	`CTRL-x CTRL-t`：根据 `thesaurus` 选项（同义词）补全单词
	-	`CTRL-x CTRL-i`：在当前文件、头文件中补全关键字
	-	`CTRL-x CTRL-]`：在当前文件、头文件中补全标签
		-	标签：可包含字母字符、`iskeyword` 选项决定字符
	-	`CTRL-x CTRL-f`：补全文件名
		-	文件名：可包含字母字符、`isfname` 选项决定字符
	-	`CTRL-x CTRL-d`：补全定义、宏
	-	`CTRL-x CTRL-v`：补全 *Vim* 命令，包括 *Ex* 命令的参数
		-	可用 `CTRL-x CTRL-q` 代替
	-	`CTRL-x CTRL-u`：通过 `completefunc` 选项自定义函数补全
	-	`CTRL-x CTRL-o`：通过 `ominifunc` 选项自定义函数补全，通常用于特定文件类型的补全
	-	`CTRL-x [CTRL-]s`：单词拼写补全
	-	`CTRL-n`、`CTRL-p`：从 `complete` 选项给出位置正向、反向补全单词

-	插入补全弹出菜单
	-	菜单可能处于 3 个状态：一般处于 1、3 状态，即完整匹配、增删后重新匹配
		-	插入完整匹配，如：`CTRL-n`、`CTRL-p` 后
		-	用光标键选择匹配项，此时不插入仅高亮
		-	插入部分匹配文本，且输入字符、或退格，此时匹配项列表根据光标前内容调整
	-	全部 3 个状态下，可以使用按键
		-	`CTRL-y`：是，接受当前匹配项并停止补全、
		-	`CTRL-e`：结束补全，回到匹配前原有内容
		-	`<PageUp>`、`<PageDown>`：反向、正向若干项后选择匹配项，不插入
		-	`<Up>`、`<Down>`：选择前、后个匹配项，同 `CTRL-p`、`CTRL-n`，不插入
		-	`<Space>`、`<Tab>`：停止补全，不改变匹配，插入键入字符
		-	`<Enter>`：状态 1、3 时插入现有文本、换行符；状态 2 时插入选择项
	-	状态 1 下
		-	`<BS>`、`CTRL-h`：删除字符，重新查找匹配项，会减少匹配项数目
		-	其他非特殊字符：停止补全不改变匹配，插入输入字符
	-	状态 2、3 下
		-	`<BS>`、`CTRL-h`：删除字符，重新查找匹配项，会增加匹配项数目
		-	`<CTRL-l>`：从当前匹配项中增加字符，减少匹配项数量
		-	任何可显示的空白字符：插入字符，减少匹配项数量

-	相关选项
	-	`infercase`：调整匹配的大小写
	-	`complete`：决定匹配搜索的缓冲区
	-	`include` 指定如何找到含有头文件名字的行
	-	`define`：包含定义的行
	-	`path`：指定搜索头文件的位置
	-	`isfname`：文件名可包含的字符
	-	`completefunc`：`CTRL-x CTRL-u` 用户补全自定义函数
	-	`ominifunc`：`CTRL-x CTRL-o` 全能补全函数
	-	`compeletopt`：补全选项
		-	`menu`：2 条匹配以弹出菜单
		-	`menuone`：1 条匹配即弹出菜单
	-	`pumheight`：菜单最大高度，缺省整个有效空间
	-	`pumwidth`：菜单最小宽度，缺省 15 字符



