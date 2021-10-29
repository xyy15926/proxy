---
title: Bash 环境
categories:
  - Linux
  - Bash
tags:
  - Linux
  - Bash
  - Shell
  - Environment
date: 2021-08-17 15:02:20
updated: 2021-09-01 20:30:15
toc: true
mathjax: true
description: 
---

##	Shell 环境

-	Shell 执行环境要素包括
	-	打开的文件：提供给 `exec` 的重定向修改，或唤醒时继承
	-	当前工作目录：`cd`、`pushd`、`popd` 设置，或唤醒时继承
	-	文件创建模式：`umask` 设置，或从父 Shell 继承
	-	当前 *traps*：`trap` 设置
	-	Shell 参数：变量赋值、`set` 设置、或从父 Shell 继承
	-	Shell 函数：执行时定义，或从父 Shell 继承
	-	Shell 选项：唤醒时指定，或 `set` 设置
	-	Bash `shopt` 设置的选项
	-	Shell 别名：`alias` 设置
	-	进程 ID，包括：后台任务 ID、`$$`、`$PPID`

-	非函数、非 Shell 内建命令执行时，于独立的 Shell 执行环境中唤醒 Shell
	-	继承以下环境元素
		-	打开的文件
		-	当前工作目录
		-	文件创建模式
		-	`export` 标记的 Shell 变量、函数，为命令 `export` 的变量
		-	被捕捉的 *traps* 的值被继承自父 Shell
	-	若命令后接 `&`、且任务控制不活动，则命令的标准输入为空文件 `/dev/null`
	-	独立 Shell 环境中命令的执行不影响其他 Shell 执行环境

-	命令替换、`()` 执行命令、异步命令、管道触发的内建命令在子 Shell 环境执行
	-	该环境是父 Shell 环境的副本，但
		-	*traps* 被重设为继承自父 Shell 环境的值
		-	非 *POSIX* 模式下，Bash 会清除继承的 `-e` 选项
	-	子 Shell 环境中命令的执行不影响父 Shell 执行环境

###	Shell 变量

Shell 变量：Shell 自行管理、分配值的变量

-	Shell 变量往往被进程读取用于自身初始化、设置，是正常运行的条件
	-	通常是系统定义好、父 Shell 传递给子 Shell
	-	所有程序都能访问环境变量，可用作进程通信
		-	在程序中设置环境变量，供其他进程访问变量
		-	父进程为子进程设置环境变量，供其访问

####	*Bourne Shell* 变量

-	*Bourne Shell* 变量：*Bash* 中与 *Bourne Shell* 使用方式相同的变量

-	常用的 *Bourne Shell* 变量有
	-	`HOME`
	-	`CDPATH`：`cd` 查找路径，`:` 分隔
	-	`IFS`：内部词分隔符，指定字符串元素之间的切分方式
		-	默认为空格、制表、回车
	-	`OPTARG`
	-	`OPTIDX`
	-	`PATH`
	-	`PS1`
	-	`PS2`

> - <https://runebook.dev/zh-CN/docs/bash/bourne-shell-variables>

####	*Bash* 变量

-	*Bash* 变量：由 *Bash* 设置、使用，其他 Shell 通常不会对其做特殊处理

-	*Bash* 配置相关变量
	-	`BASHPID`：当前 *Bash* 进程 ID
	-	`BASHOPTS`：当前 *Bash* 参数（`shopts` 设置）
	-	`SHELL`：Shell 名称
	-	`SHELLOPTS`：当前 Shell 参数（`set` 设置）

-	环境相关变量
	-	`DISPLAY`：图形环境显示器名
		-	通常为 `:0`，表示 *X Server* 的首个显示器
	-	`TERM`：终端类型名，即终端模拟器所用协议
	-	`EDITOR`：默认文本编辑器
	-	`HOME`：用户主目录
	-	`HOST`：主机名
	-	`LANG`：字符集、语言编码
		-	一般为 `zh_CN.UTF-8`
	-	`PATH`：冒号分开的目录列表，指示可执行文件搜索地址
	-	`PWD`：当前工作目录

-	用户相关变量
	-	`UID`：当前用户 ID
	-	`USER`：当前用户名
	-	`LOGNAME`：用户登录名
	-	`LANG/LANGUAGE`：语言设置

-	特殊变量：手动修改后重置也不能恢复其行为
	-	`RANDOM`：返回 0-32767 间随机数
	-	`LINENO`：当前正在执行脚本、函数的行号

###	环境变量设置

-	环境变量的范畴
	-	Shell 变量
	-	可被子 Shell 继承的变量，即 `export` 标记的变量
	-	Shell 执行环境中的所有变量

-	`export`/`declare -x`：设置全局（环境）变量
	-	任何在该shell设置环境变量后，启动的（子）进程都会
		继承该变量
	-	对于常用、许多进程需要的环境变量应该这样设置
	
-	`<ENV_NAME>=... cmd`：设置临时环境变量
	-	`<ENV_NAME>=...`不能被子进程继承，所以必须在其后立刻接命令
	-	只对当前语句有效，**不覆盖** 同名变量

-	狭义：`export`/`declare -x`声明的变量，只有这样的变量才能默认被子进程继承
-	广义：shell中所有的变量（包括局部变量）

####	`/etc/environment`

-	`/etc/environment`：设置整个系统的环境
	-	系统在登陆时读取第一个文件
	-	用于所有为所有进程设置环境变量

##	Bash 配置项

###	`shopt `

-	`shopt`：调整 *Bash* 行为
	-	`$ shopt -s <option-name>`：打开参数
	-	`$ shopt -u <option-name>`：关闭参数
	-	`$ shopt <option-name>`：查询参数状态

-	选项
	-	`dotglob`：模式扩展结果包含隐藏文件
	-	`nullglob`：文件名扩展不存在符合条件文件时返回空（整体返回空）
	-	`extglob`：使得 *Bash* 支持 *ksh* 的部分模式扩展语法（主要是量词语法）
	-	`nocaseglob`：文件名模式扩展不区分大小写（整体不区分）
	-	`globstar`：`**` 匹配 0 个或多个子目录

###	`set`

-	`set`：设置所使用的 Shell 选项、列出 Shell 变量
	-	缺省显示 **全部**  Shell 变量、函数
	-	`$ set -<op>`：根据选项标志 `op` 设置 Shell
	-	`$ set +<op>`：取消选项标志 `op`
	-	`$ set -o <option-name>`：根据选项名 `option-name` 设置 Shell
	-	`$ set +o <option-name>`：取消选项名 `option-name`
	-	`$ set -- <pos-params>`：设置位置参数为 `pos-params`

-	选项参数
	-	`-a`：输出之后所有至 `export`（环境变量）
	-	`-b`：使被终止后台程序立即汇报执行状态
	-	`-B`：执行括号扩展
	-	`-C`：重定向所产生的文件无法覆盖已存在文件
	-	`-d`： Shell 默认使用 hash 表记忆已使用过的命令以加速执行，此设置取消该行为
	-	`-e`：若指令回传值不为 0，立即退出 Shell 
	-	`-f`：取消模式扩展
	-	`-h`：寻找命令时记录其位置???
	-	`-H`：（默认）允许使用 `!` 加 *<编号>*方式执行 `history` 中记录的命令
	-	`-k`：命令后的 `=` 赋值语句，也被视为设置命令的环境变量
	-	`-m`：监视器模式，启动任务控制
		-	后台进程已单独进程组运行
		-	每次完成任务时显示包含退出的状态行
	-	`-n`：读取命令但不执行
		-	通常用于检查脚本句法错误
	-	`-p`：允许 *set-user/group-id*
		-	禁止处理 `$ENV` 文件、从文件中继承 Shell 函数
	-	`-P`：处理 `cd` 等改变当前目录的命令时，不解析符号链接
	-	`-t`：读取、执行下条命令后退出
	-	`-u`：使用未设置变量作为错误处理
	-	`-v`：输入行被读取时，显示 Shell 输出行
	-	`-x`：执行命令前先输出执行的命令、环境变量
		-	命令文本前导符为 `$PS4` 扩展值

-	`-o` 选项参数为下列之一
	-	`allexport`：同`-a`
	-	`braceexpand shell`：（默认）执行花括号扩展
	-	`emacs`：（默认）使用emacs风格命令行编辑接口
	-	`errexit`：同`-e`
	-	`errtrace`：同`-E`
	-	`functrace`：同`-T`
	-	`hashall`：同`-h`
	-	`histexpand`：同`-H`
	-	`history`：记录命令历史
	-	`ignoreeof`：读取EOF时不退出shell
	-	`interactive-comments`：允许交互式命令中出现注释
	-	`keyword`：同`-k`
	-	`monitor`：同`-m`
	-	`noclobber`：同`-C`
	-	`noexec`：同`-n`
	-	`noglob`：同 `-f`
	-	`nohash`：同 `-d`
	-	`notify`：同 `-b`
	-	`nounset`：同 `-u`
	-	`physical`：同 `-P`
	-	`pipfail`：管道命令返回值为最后返回值非 0 命令的状态，若没有非 0 返回值返回 0
	-	`posix`：改变 shell 属性以匹配标准，默认操作不同于 *POSIX1003.2* 标准
	-	`priviledged`：同 `-p`
	-	`verbose`：同 `-v`
	-	`vi`：使用 vi 风格的命令编辑接口
	-	`xtrace`：同 `-x`

> - `$-`中存放有当前已设置标志位

###	`/usr/bin/env`

-	`env`：在修改过的环境中执行命令
	-	`$ env <op> <name>=<value> <cmd>`：在设置变量 `name` 的环境中执行 `cmd`
	-	`$ env`：打印当前环境中环境变量

-	选项参数
	-	`<NAME>`：在 `PATH` 中查找 `NAME` 命令并执行，缺省打印 Shell 中环境变量
	-	`-i`/`--ignore-environment`：不带环境变量启动
	-	`-u`/`--unset=<NAME>`：从环境变量中删除变量

-	说明
	-	`env` 常用脚本的 *shebang* 行
	-	`env` 本质上是调用 `execve` 修改子进程环境变量

###	`export`

-	`export`：标记可以传递给子进程的变量
	-	`$ export <op> [<name>[=<val>]]$`：可以赋值语句作为参数

-	选项参数
	-	`-f`：标记函数
	-	`-n`：不再标记每个变量可传递给子进程
	-	`-p`：打印标记为可传递给子进程的变量

##	*Bash* 配置

###	*Prompt* 控制

-	`$PROMPT_COMMANDS`：*Bash* 输出 prompt 前按顺序执行其中命令

-	prompt 控制变量
	-	`$PS0`：读取、执行命令提示
	-	`$PS1`：主提示字符串
	-	`$PS2`：次提示字符串，折行输入提示符
		-	缺省为 `>`
	-	`$PS3`：`select` 命令提示
	-	`$PS4`：`-x` 标志时，命令回显提示
		-	缺省为 `+`

-	可扩展特殊字符
	-	`\a`：响铃
	-	`\d`：日期
	-	`\D{<format>}`：`format` 作为参数传递给 `strftime`
	-	`\e`：转义字符
	-	`\h`：主机名
	-	`\H`：主机名
	-	`\j`：当前 Shell 管理的任务数量
	-	`\l`：Shell 所处终端设备名称的 basename
	-	`\n`：新行
	-	`\r`：回车
	-	`\s`：Shell 名，即 `$0` 的 basename
	-	`\t`：时间，24小时 `HH:MM:SS` 格式
	-	`\T`：时间，12小时 `HH:MM:SS` 格式
	-	`\@`：时间，12小时 *am/pm* 格式
	-	`\A`：时间，24小时 `HH:MM` 格式
	-	`\u`：用户名
	-	`\v`：Bash 版本
	-	`\V`：Bash release
	-	`\w`：当前工作目录完整路径，`$HOME` 被替换为 `~`
		-	`$PROMPT_DIRTRIM` 控制展示目录层数，缺省为 0
	-	`\W`：`$PWD` 的 basename，`$HOME` 替换为 `~`
	-	`\!`：命令的历史序号（计入从历史记录中恢复命令）
	-	`\#`：命令的命令号（不计入从历史记录中恢复命令，仅计入当前 Shell 会话中执行命令）
	-	`\$`：若有效 *uid* 为 0（即 `root` 用户）则为 `#`，否则为 `$`
	-	`\<nnn>`：八进制 `nnn` 对应 *ASCII* 字符
	-	`\\`：反斜杠
	-	`\[`：不可打印字符序列起始，常用于引导终端控制字符（*ANSI* 转义序列）
	-	`\]`：不可打印字符序列结束

> - <https://runebook.dev/en/docs/bash/controlling-the-prompt>


