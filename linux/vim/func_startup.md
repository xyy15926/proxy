---
title: *Vim* 启动、初始化
categories:
  - Linux
  - Tool
  - Vi
tags:
  - Linux
  - Tool
  - Vi
  - Editor
date: 2021-11-03 11:11:23
updated: 2021-12-06 15:52:08
toc: true
mathjax: true
description: 
---

##	*Vim* 启动

|Alias|等价 *Vim* 启动选项|含义|
|-----|-----|-----|
|`$ ex`|`$ vim -e`|*Ex* 模式|
|`$ exim`|`$ vim -E`|增强的 *Ex* 模式|
|`$ view`|`$ vim -R`|只读|
|`$ gvim`|`$ vim -g`|*GUI* 启动|
|`$ gex`|`$ vim -eg`|*Ex* 模式启动 *GUI*|
|`$ gview`|`$ vim -Rg`|*GUI* 启动只读|
|`$ rvim`|`$ vim -Z`|受限模式|
|`$ rview`|`$ vim -RZ`|受限模式 `view`|
|`$ rgvim`|`$ vim -gZ`|受限模式 `gview`|
|`$ rgview`|`$ vim -RgZ`|受限模式 `gview`|
|`$ evim`|`$ vim -y`|简易 *Vim*，置位 `insertmode`|
|`$ eview`|`$ vim -yR`|只读 `evim`|
|`$ vimdiff`|`$ vim -d`|比较模式|
|`$ gvimdiff`|`$ vim -gd`|*GUI* 启动比较模式|

-	开启编辑时，以下 5 种参数仅必选其一
	-	`<FILE>`：编辑已有文件，读入缓冲区，首个文件作为当前文件
	-	`-`：效果取决于是否使用 *Ex* 模式
		-	`$ vim -` / `$ ex -v -`：普通模式，从标准输入读取文本
		-	`$ ex -` / `$ vim -e -` / `$ evim -` / `$ vim -E`：*Ex* 模式，安静模式开始编辑
	-	`-t <TAG>`：从标签文件中查找 `TAG`，以相关文件作为当前文件，并执行相关命令
		-	通常用于 *C* 程序，查找、定位 `TAG`
	-	`-q [ERROR_FILE]`：快速修复模式，读入错误文件并显示第一个错误
		-	`ERROR_FILE` 缺省为 `error_file` 选项值
	-	空白：编辑新的空白、无名称缓冲区

-	部分选项参数
	-	`-r [FILENAME]`：常看交换文件（特定文件对应交换文件）
	-	`-R [FILENAME]`：只读方式打开文件
	-	`-S [SESSION_FILE]`：打开会话文件

###	*EVim*

-	*EVim* 是设置了 *点击-输入* 风格的 *Vim*
	-	便于新手使用键鼠操作
	-	任何按键均将进入 `insertmode`，即总是可以直接编辑
	-	`<C-L>` 可以推出此模式

> - <https://yianwillis.github.io/vimcdoc/doc/starting.html>

###	*Ex* 模式

-	*Ex* 模式：以 *Ex* 命令进行编辑操作
	-	仅可使用 *Ex* 命令进行编辑
	-	缓冲区渲染结果不再随输入更新
	-	可在 *Normal* 模式下以 `Q` 命令进入

###	*Vim* 初始化

####	初始化流程

> - 此 *Vim* 初始化顺序为非 *GUI* 版本

-	设置 `shell`、`term` 选项
	-	`shell`：优先使用环境变量 `$SHELL` 设置（*Win32* 上使用 `COMSPEC`）
	-	`term`：优先使用环境变量 `$TERM` 设置

-	处理参数：检查命令行上给出的选项、文件名
	-	为所有文件创建缓冲区（还未载入文件）
	-	`-V`：可显示、记录执行情况，方便调试

-	从初始化文件、环境变量执行 *Ex* 命令
	-	载入 `$VIMRUNTIME/evim.vim`：仅 *Vim* 以 `evim`、`eview` 模式启动
	-	载入系统 `vimrc` 文件：`$VIM/vimrc`
		-	总是按 `compatible` 模式载入（选项还未设置）
	-	载入用户 `vimrc` 文件、*vim* 环境变量：仅首个被找到的文件、环境被执行
		-	`$VIMINIT` 环境变量
			-	多个命令用 `|`、`<NL>` 分隔
		-	用户 *vimrc* 文件：`$ vim --version` 查看详细优先级
			-	*Unix*：`$HOME/.vimrc`、`$HOME/.vim/vimrc`
		-	`$EXINIT` 环境变量
		-	用户 `exrc` 文件
		-	默认 `vimrc` 文件：`$VIMRUNTIME/defaults.vim`
	-	搜索当前目录 `vimrc` 文件：仅在 `exrc` 选项被设置时
		-	`.vimrc`、`_vimrc`
		-	`.exrc`、`_exrc`
	-	说明事项
		-	可用 `-u` 指定初始化文件，此时以上 3 步初始化均被跳过，另外
			-	`-u NORC`：跳过此初始化
			-	`-u NONE`：跳过插件载入
		-	`vimrc` 文件：包含初始化命令，每行作为 *Ex* 命令执行
		-	`vimrc` 是 *Vim* 专用名称，`exrc` 被 *Vi* 使用

-	载入插件脚本：相当于执行命令 `:runtime! plugin/**/*.vim`
	-	按以下顺序载入插件脚本
		-	插件脚本：`runtimepath` 选项值中除 `after` 结尾目录下 `plugin/**/*.vim` 文件
			-	按 `runtimepath` 中顺序依次搜索，但跳过以 `after` 结尾的目录
			-	递归搜索、载入全部 `.vim` 脚本
		-	插件包：`packpath` 选项值中各目录下 `pack/*/start/*/plugin/**/*.vim` 文件
			-	事实上，在上步载入插件脚本前会先查找、记录插件包中插件：`packpath` 各目录下 `pack/*/start/*`
			-	插件脚本执行完毕后，`pack/*/start/*`、`pack/*/start/*/after` 添加进 `runtimepath`
		-	`after` 插件：新 `runtimepath` 中 `after` 结尾目录下 `plugin/**/*.vim` 文件
			-	以上 2 步后，插件包中 `after` 被加入 `runtimepath`，同一逻辑中被载入
			-	事实上，软件包 `after` 位置更靠前，较先执行
	-	以下情况将不载入插件脚本
		-	`loadplugins` 选项复位
			-	命令行 `-c 'set noloadplugins'` 不生效，此时命令行命令未执行
			-	命令行 `--cmd 'set noloadplugins'` 生效
		-	`--noplugins` 命令行参数
		-	`--clean` 命令行参数
		-	`-u NONE` 命令行参数
		-	*Vim* 无 `+eval` 特性

-	启动前命令行参数、选项设置处理
	-	设置选项 `shellpipe`、`shellredir`
		-	根据 `shell` 选项值设置，除非已设置
	-	命令行参数 `-n`：设置 `updatecount` 选项为 0
	-	命令行参数 `-b`：置位 `binary` 选项
	-	*GUI* 初始化
	-	`viminfo` 选项非空：读入 *viminfo* 文件
	-	命令行参数 `-q`：读入快速修复文件，失败则 *Vim* 退出

-	打开窗口
	-	命令行标志 `-o`：打开所有窗口
	-	命令行标志 `-p`：打开所有标签页
	-	切换屏幕，启动
	-	命令行标志 `-q`：跳到首个错误
	-	载入所有窗口的缓冲区，不触发 `BufAdd` 自动命令

-	执行启动命令
	-	命令行标志 `-t`：跳转至标签处
	-	命令行 `-c` 、`+cmd`：执行给出命令
	-	`insertmode` 选项置位：进入插入模式
	-	复位启动标志位，`has("vim_starting")` 返回 0
	-	`v:vim_did_enter` 变量设为 1
	-	执行 `VimEnter` 自动命令

> - `-s` 参数将跳过前述 4 步初始化，仅 `-u` 生效

####	流程说明

-	*Vi* 兼容性问题
	-	*Vi* 启动时， `compatible` 选项默认置位，初始化时使用该设置
		-	`vimrc` 总是以 `compatible` 模式读入，选项复位在其后发生
	-	以下情况下存在会复位 `compatible` 选项
		-	找到用户 `vimrc` 文件、`$VIMINIT` 环境变量、当前目录下 `vimrc` 文件、`gvimrc` 文件
		-	`-N`、`--clean` 命令行参数
		-	载入 `defaults.vim` 脚本

-	默认用户 `vimrc`：`$VIMRUNTIME/defaults.vim`
	-	置位 `skip_defaults_vim` 变量可避免执行
	-	若确定需要执行，最好 `unlet! skip_defaults_vim` 后再 `source`

###	*Vim Runtime*

> - 以 *Vim82* 为基准

####	*Runtime* 目录结构

-	`$VIMRUNTIME` 目录结构：仅包含涉及初始化关键目录、文件
	-	`vimrc`：系统初始化文件，其中
		-	`:syntax on`
		-	`:filetype on`
	-	`syntax/`：语法高亮目录
		-	`syntax/syntax.vim`：其中
			-	`:runtime syntax/synload.vim`
		-	`syntax/synload.vim`：其中
			-	`:runtime syntax/<FILETYPE>.vim`
			-	`:runtime syntax/syncolor.vim`
		-	`syntax/syncolor.vim`：
	-	`filetype.vim`、`ftoff.vim`：文件类型探测，其中
		-	`:runtime ftdetect/*.vim`
		-	`:runtime scripts.vim`
	-	`indent.vim`、`indoff.vim`：文件类型缩进，其中
		-	`:runtime indent/<FILETYPE>.vim`
	-	`ftplugin.vim`、`ftplugof.vim`：文件类型插件，其中
		-	`:runtime ftplugin/<FILETYPE>.vim`
	-	`autoload/`：自动载入脚本
		-	`:call <FILENAME>#<FUNC>` 自动载入 `autoload/FILENAME`
	-	`colors/`：颜色配置脚本
		-	`:colorscheme <FILENAME>` 载入 `colors/<FILENAME>` 中配色
	-	`plugin/`：插件脚本目录，初始化时自动载入
	-	`default.vim`：缺省用户初始化文件

-	初始 `runtimepath` 中 *runtime* 目录结构
	-	`vimrc`：用户初始化温婉
	-	`plugin/`：插件脚本目录，初始化时自动载入
	-	`autoload/`：自动载入脚本
	-	`colors/`：颜色配置脚本
	-	`pack/`：插件包目录
	-	被命令触发
		-	`filetype.vim`、`ftoff.vim`
		-	`ftplugin.vim`、`ftplugof.vim`
		-	`indent.vim`、`indoff.vim`
	-	`autocmd` 关联文件类型触发
		-	`syntax/*.vim`
		-	`ftplugin/*.vim`
		-	`indent/*.vim`

-	相关命令：仅涉及脚本触发方式部分
	-	`syntax [on|off|enable|manual]`：语法设置
		-	`on`/`enable`：打开语法，载入 `$VIMRUNTIME/syntax/syntax.vim`
		-	`off`：关闭语法，载入 `$VIMRUNTIME/syntax/nosyntax.vim`
		-	`manual`：手动语法，载入 `$VIMRUNTIME/syntax/manual.vim`
		-	`clear`：清除当前缓冲区语法配置
	-	`filetype [indent] [plugin] [on|off]`：文件类型探测、缩进、插件设置
		-	`indent`：文件类型缩进
			-	开启时：载入 `runtimepath/indent.vim`
			-	关闭时：载入 `runtimepath/indentoff.vim`
		-	`plugin`：文件类型插件
			-	开启时：载入 `runtimepath/ftplugin.vim`
			-	关闭时：载入 `runtimepath/ftplugof.vim`
		-	`on`：开启，可认为总是省略开启探测
			-	即包含 `on` 时总是会开启文件探测
			-	开启文件探测：载入 `runtimepath/filetype.vim`
		-	`off`：关闭
			-	仅在 `filetype off` 才关闭文件探测
			-	关闭文件探测：载入 `runtimepath/ftoff.vim`

####	`pack/` 插件包、`after/`

-	（以）`after/`（结尾）：延迟处理脚本目录
	-	目录结构应类似普通 *runtimepath* 目录
	-	需位于 `runtimepath` 才生效
		-	*Vim* 启动时，延迟处理其中插件 `plugin`
		-	*Vim* 启动后，和其他目录角色、功能一致
	-	但其中不能包含 `pack/`，会崩溃

-	`pack/`：存放插件包目录，应该位于 `packpath` 中目录下
	-	插件包 *bundle*/*package*：包含一个、多个插件的目录
		-	统一管理，多个插件间可包含依赖关系
		-	避免和其他插件的文件混杂
		-	方便更新
	-	插件包 `pack/<PACKAGE_1>` **内** 目录结构
		-	`start/`：启动时在 `runtimepath/plugin` 加载后自动加载的插件
			-	`<PLUGIN_1>/`
			-	`<PLUGIN_2>/`
		-	`opt/`：启动后 `:packadd <PLUGIN_NAME>` 手动加载的插件
			-	`<PLUGIN_EXTRA_1>/`
			-	`<PLUGIN_EXTRA_2>/`

-	`pack/<PACKAGE_1>/start/<PLUGIN_1>`：插件目录
	-	目录结构应类似普通 *runtimepath* 目录
	-	*Vim* 启动时
		-	类似 `runtimepath` 其他目录，其中 `plugin/**/*.vim` 被载入
		-	目录自身、其中 `after/` 目录被添加进 `runtimepath`
	-	*Vim* 启动后，和 `runtimepath` 中其他目录角色、功能一致

-	`pack/<PACKAGE_1>/opt/<PLUGIN_1>` 类似
	-	但不在 *Vim* 启动时载入，需手动 `:packadd` 载入插件
	-	插件目录、`/after` 加入 `runtimepath` 首、尾

	> - 事实上，插件中 `ftdetect/*.vim` 会在 `plugin/**/*.vim` 后被载入，而其中 `filetype.vim` 不被载入

###	`$VIM`、`$VIMRUNTIME`

-	`$VIM` 用于定位 *Vim* 使用的用户文件，按照如下顺序取值
	-	`$VIM` 环境变量
	-	`helpfile` 选项值（若其中不包含其他环境变量）确定的目录
	-	*Unix*：编译时定义的安装目录
	-	*Win32*：可执行文件的目录名确定的目录

-	`$VIMRUNTIME` 用于定位支持文件，如：帮助文档、语法高亮文件，按如下顺序取值
	-	`$VIMRUNTIME` 环境变量
	-	`$VIM/vim{VERSION}`
	-	`$VIM/runtime`
	-	`$VIM`
	-	`helpfile` 选项值确定的目录

> - 可通过 `:let $VIM=`、`:let $VIMRUNTIME=` 修改二者

##	*Vim* 相关功能

###	暂停

-	暂停命令、快捷键
	-	`:st[op][!]`、`:sus[pend][!]`：暂停 *Vim*
		-	无 `!` 且置位 `autowrite`，每个修改过、由文件名的缓冲区被写回
		-	`!` 或 `autowrite` 未置位，则修改过的缓冲区不被写回
	-	`<C-z>`：类似 `:stop` 命令（也即 *Unix* 系统一般的挂起快捷键）

-	禁止处理输入
	-	`<C-s>`：阻止 *Vim* 处理输入（中间输入均被记录，恢复后被处理）
	-	`<C-q>`：恢复 *Vim* 处理输入

###	保存配置

-	配置保存相关命令
	-	`:mk[exrc][!] [FILE]`：写入当前键盘映射、修改过的选项至 `FILE`
		-	`FILE` 缺省为 `./.exrc`
		-	`!`：允许覆盖已有的文件
	-	`:mkv[imrc][!] [FILE]`：类似 `:mkexrc`，同时写入 `:version`
		-	`FILE` 缺省为 `./.vimrc`

-	配置保存命令会将 `:set`、`:map` 命令写入文件
	-	部分和终端、文件有关的配置不被保存
	-	只有保存全局映射，局部于缓冲区的映射被忽略

###	保存视图

-	视图：应用于一个窗口的设置的集合
	-	可以保存视图并在之后恢复
		-	恢复文本显示方式
		-	恢复窗口的选项、映射
	-	视图保存的内容：可选项由 `viewoptions` 指定
		-	窗口使用参数列表
		-	窗口编辑的文件
		-	光标位置

-	视图保存相关选项
	-	`viewdir`：视图存储目录，缺省为 `$VIM/view`
	-	`viewoptions`：指定视图保存、恢复内容的列表
		-	`options`、`localoptions`：恢复映射、缩写、局部选项的局部值
		-	`folds`：恢复折叠
		-	`curdir`：恢复当前目录

-	视图保存相关命令
	-	`:mkv[iew][!] [FILE]`：生成 *Vim* 脚本，用于恢复当前窗口内容
		-	`FILE` 不提供或提供 1-9 数字时，自动生成名称，并存放在 `viewdir` 选项指定的目录下
			（即允许保存同一文件 10 个视图）
		-	`!` 允许覆盖已有文件
	-	`:lo[adview] [FILE]`：从文件中恢复视图

###	保存会话

-	会话：所有窗口的视图、全局设置
	-	可以保存会话并在之后恢复
		-	恢复窗口布局
		-	恢复窗口视图
		-	建立多个会话以在不同项目间快速切换
	-	在 `vim` 内执行 `:source <SESS_FILE>` 即可恢复
		-	关闭当前窗口
			-	关闭当前标签页处当前窗口以外窗口
			-	关闭除当前标签页外其他标签：可能导致缓冲区卸载
			-	若当前缓冲区为无名且空，删除
		-	恢复会话：可选项由 `sessionoptions` 选项指定
			-	恢复缓冲区列表
			-	恢复光标位置
			-	恢复各窗口视图
		-	与会话文件同名，但以 `x.vim` 结尾文件同样被执行
			-	可用于给指定会话的附加设置、动作

-	会话保存相关命令
	-	`:mks[ession]! [FILE]`：生成 *Vim* 脚本，用于恢复当前会话
		-	`FILE` 缺省为 `Session.vim`
		-	`!`：允许覆盖已有文件
		-	保存内容由 `sessionoptions` 选项决定

-	会话保存相关选项
	-	`:sessionoptions`：决定会话保存、恢复的内容的列表
		-	`options`：恢复全局映射、选项（局部选项的全局值）
		-	`globals`：恢复大写字母开始、至少包含一个小写字母的全局变量
		-	`curdir`：恢复当前目录
		-	`sesdir`：设置当前目录为会话文件所在位置
		-	`winpos`：恢复 *GUI* 窗口位置
		-	`resize`：恢复屏幕大小
		-	`buffers`：恢复所有缓冲区，包括隐藏、未载入，否则仅打开缓冲区
		-	`help`：恢复帮助窗口
		-	`blank`：恢复编辑无名缓冲区的窗口
		-	`winsize`：若无任何窗口舍弃，则恢复窗口大小
		-	`tabpages`：包含所有标签页

-	其他
	-	`SessionLoadPost`：会话文件载入、执行后激活的自动命令事件
	-	`v:this_session`：保存会话的完整文件名

> - 须编译时启用 `+mksession` 特性
> - 可通过 `-S <SESS_FILE>` 启用保存的会话文件快速启动编辑

###	*viminfo* 文件

-	*viminfo* 文件：记住所有视图、会话都使用的信息，允许继续上次退出的编辑
	-	保存内容
		-	命令行历史
		-	搜索字符串历史
		-	输入行历史
		-	非空寄存器历史
		-	多个文件的位置标记
		-	文件标记：指向文件位置
		-	最近搜索、替换模式
		-	缓冲区列表
		-	全局变量
	-	*viminfo* 文件不依赖工作内容
		-	通常只有一个 *viminfo* 文件
	-	*viminfo* 文件默认被设置为不能被其他用户读取，避免泄露可能包含文本、命令
		-	每次 `vim` 替换其时会保留被用户主动更改的权限
		-	*Vim* 不会覆盖当前实际用户不能写入的 *viminfo* 文件，避免 `$ su root` 生成无法读取的文件
		-	*viminfo* 文件不能是符号链接

> - 须编译时启用 `+viminfo` 特性

####	相关选项、命令

-	相关选项
	-	`viminfofile`：指定 *viminfo* 文件名
		-	`NONE`：不读写任何 *viminfo* 文件
		-	*Unix* 上缺省为 `$HOME//.viminfo`，*Win32* 缺省为 `$HOME/_viminfo`、`$VIM/_viminfo`
	-	`viminfo`：设置 *viminfo* 保存内容、数量限制标志列表
		-	`r`：指定不需要保存位置标记的文件
		-	`n`：指定另一个 *viminfo* 文件名
		-	`c`：若 *viminfo* 文件编码与当前 `encoding` 选项不同，尝试转换其编码后读取

-	相关命令
	-	`:rv[iminfo][!] [FILE]`：读取 *viminfo* 文件
		-	`!`：允许覆盖已经有值的信息：寄存器、位置标记等
	-	`:wv[iminfo][!] [FILE]`：写信息至 *viminfo* 文件
		-	`!` 不读原 *viminfo* 信息，只写入系统内部信息
	-	`:ol[files]`：列出 *viminfo* 文件中有存储位置标记的文件列表
		-	在启动时读入，只有只有 `:rv!` 可以改变
	-	`:bro[wse] ol[files][!]`：类似 `:ol` 列出文件名，之后可输入编号编辑指定文件
		-	`!`：放弃已修改缓冲区

####	*viminfo* 文件处理逻辑

-	若启动时 `viminfo` 选项非空，*viminfo* 文件内容被读入
	-	其中信息在适当地方被应用
		-	`v:oldfiles` 变量被填充值
			-	有位置标记的文件名
		-	启动时不读入位置标记
	-	*viminfo* 文件可以手动修改
	-	若读入 *viminfo* 文件时检查到错误
		-	之后不会覆盖该文件
		-	超过 10 个错误则停止

-	若退出时 `viminfo` 选项非空，相关信息保存在 *viminfo* 文件中
	-	保存内容、数量限制由 `viminfo` 选项定义
	-	多数选项：保存当前会话中被改变的值，未变动的值则从元 *viminfo* 文件中填充
	-	部分选项：使用时间戳保留最近改动版本，总是保留最新项目
		-	命令行历史
		-	搜索字符串历史
		-	输入行历史
		-	非空寄存器内容
		-	跳转表
		-	文件标记

####	*viminfo* 文件保存内容说明

-	位置标记：可为每个文件存储位置标记
	-	位置标记只在退出 *Vim* 时保存
	-	若须 `:bdel` 清除缓冲区，可以手动使用 `:wv` 保存位置标记
		-	`[`、`]` 不被保存
		-	`"` 、`A-Z` 被保存
		-	`0` 会被设置为当前光标，并依次覆盖之后数字位置标记

###	帮助

-	`:help [THEME]`：获得特定主题的帮助，缺省显示总览帮助窗口
	-	`THEME` 帮助主题可以是命令、功能、命令行参数、选项
		-	包含控制字符命令：控制字符前缀 `:help CTRL-A`
		-	特殊按键：尖括号 `:help i_<Up>`
		-	不同模式下命令：模式前缀 `:help i_CTRL-A`
			-	普通模式：无前缀
			-	可视模式：`v_`
			-	插入模式：`i_`
			-	命令行编辑、参数：`c_`
			-	*Ex* 命令：`:`
			-	用于调试的命令：`>`
			-	正则表达式：`/`
		-	命令行参数：横杠 `:help -t`
		-	选项：引号括起 `:help 'number'`

-	`:helpgrep [THEME]`：在所有帮助页面中搜索（包括已安装插件）

##	其他CMD命令

-	`q:`：vim命令历史窗口
> - `|` 管道符可以用于行内分隔多个命令 `:echom "bar" | echom "foo"`

###	编辑过程中使用shell

-	`:!{shell command}`即可直接执行shell命令并暂时跳出vim
	-	`:r !{shell command}`可以将输出结果读取到当前编辑
	-	`:w !{sheel command}`可以将输出结果输出到vim命令行

-	`:shell`即可暂时进入到shell环境中，`$exit`即可回到vim中

-	`<c-z>`暂时后台挂起vim回到shell环境中，`$fg`即可回到之前
	挂起的进程（此时为vim，详见fg命令）

###	读取结果

-	`:read cmd`：读取`cmd`结果至光标下一行
	-	`:read !date`：读取系统`date`命令结果

###	交换文件处理

-	确认需要恢复：直接恢复`R`

-	确认丢弃：直接删除`D`

-	没有明确目标：只读打开`O`

-	比较交换文件、现在文件差别

	-	恢复打开`R`
	-	另存为其他文件`:saveas filname.bak`
	-	与当前文件对比`:diffsplit filename`

-	一般不会直接`E`（*edit anyway*）

###	外部功能

-	相关快捷键
	-	`K`：对当前单词调用`keywordprg`设置的外部程序，默认“man”





