---
title: Bash 使用
categories:
  - Linux
  - Bash
tags:
  - Linux
  - Bash
  - Shell
  - Config
date: 2021-08-17 14:54:40
updated: 2021-08-25 23:09:08
toc: true
mathjax: true
description: 
---

##	Shell 介绍

![pc_architecture](imgs/pc_architecture.png)

*Shell*：提供 **用户界面** 的程序，接受用户输入命令和内核沟通

-	*Shell* 向用户提供操作系统入口
	-	避免普通用户随意操作导致系统崩溃
	-	虽然Shell会调用其他应用程序，其他应用程序再调用系统调用，而不是直接与内核交互

-	*Command-Line Interface*：命令行 *Shell*，通常不支持鼠标，通过键盘输入指令
	-	`sh`：*Bourne Shell*
	-	`ash`：*Almquist Shell*
		-	*Bourne Shell* 的简化版本
		-	*NetBSD* 的默认 Shell
	-	`bash`：*Bourne-Again Shell*
		-	最广泛使用交互 Shell，即 `$SHELL`
	-	`dash`：*Debian Almquist Shell*
		-	`ash` 的改进版本，作为 *Debian* 系发行版脚本执行 Shell，即 `/bin/sh`
		-	功能较 `bash` 少得多、速度更快，不适合用于交互
		-	语法严格遵守 *POSIX* 标准
	-	`zhs`：*Z Shell*
	-	`fish`：*Friendly Interactive Shell*
	-	`cmd.exe`
		-	其宿主作为隐式 *Terminal Emulator*
	-	`PowerShell`

-	*GUI*：*Graphic User Interface* 图形 *Shell*
	-	*Windows* 下的 *explorer.exe*

> - 大部分 Shell 都兼容 *POSIX* Shell 协议（除 `cmd.exe`），但有自定义的命令、关键字
> - <https://runebook.dev/zh-CN/docs/bash/-index-#Manual>

###	*Shell* 和 *Terminal*

-	*Shell* 更多指提供和内核交互入口的软件，提供
	-	命令提示符 *Prompt*
	-	行编辑、输入历史、自动补全（但是也有些终端自己实现此功能）

-	*Terminal* 更多指 *IO* 端口硬件（驱动），提供
	-	用户输入、输出交互支持
	-	上、下翻页查看内容
	-	终端中复制、粘贴功能

##	行操作

-	*Bash* 中内置有 *realine* 库，支持其提供很多 *行操作* 功能
	-	库默认使用 `emacs` 风格快捷键
		-	直接在 *Bash* 设置 `$ set -o vi`、`$ set -o emacs`
		-	在 *readline* 配置文件中指定 `set editing-mode vi`
	-	*Bash* `--noediting` 选项指定不带 *readline* 库启动 *Bash*

> - <https://wangdoc.com/bash/readline.html>

###	窗口内容

-	清屏
	-	`C-l`：清除屏幕并将当前行移到顶部，同 `$ clear`
	-	`<S-PageUp>`：向上滚动
	-	`<S-PageDown>`：向下滚动

-	光标移动
	-	`C-a`：移至行首
	-	`C-e`：移至行尾
	-	`C-b`：向行首移动一字符，同左箭头
	-	`C-f`：向行尾移动一字符，同右箭头
	-	`A-f`/`Esc-f`：移动到当前词尾
	-	`A-b`/`Esc-b`：移动到当前词首

-	编辑操作
	-	`C+d`：删除光标处字符（若光标处无任何字符，会退出 Shell）
	-	`C+w`：删除光标前单词
	-	`C+t`：交换光标处、前一字符位置
	-	`A+t`/`Esc+l`：交换光标处、前一词位置
	-	`A+l`/`Esc+l`：转换光标位置至词尾为小写
	-	`A+u`/`Esc+l`：转换光标位置至词尾为大写

-	剪切+粘贴
	-	`C+k`：剪切光标处至行尾
	-	`C+u`：剪切光标处至行首
	-	`A+d`/`Esc+d`：剪切光标处至词尾
	-	`A+Backspace`/`Esc+Backspace`：剪切光标处至词首
	-	`C+y`：在光标处粘贴

###	任务管理

-	`<C-c>`：终止当前任务
-	`<C-d>`：关闭 Shell 会话

###	自动补全

-	自动补全
	-	`Tab`：候选唯一按一下直接补全，否则需要按两下给出候选
		-	缺省补全路径、命令
		-	`$` 开头：补全变量
		-	`~` 开头：补全用户名
		-	`@` 开头：补全主机名

##	历史记录

###	浏览、查找历史记录

> - 退出 *Bash* 时，操作历史将被写入 `HISFILE` 变量指定的文件中

-	`Up`、`Down`：浏览已执行命令的历史记录
-	`C-p`、`C-n`：同 `Up`、`Down`
-	`A+<`：第一个命运
-	`A+>`：最后一个命运
-	`C-r`：搜索操作历史，查询、显示最近的匹配结果

###	历史记录引用

-	命令历史记录引用
	-	`!!`：引用上条命令
	-	`!<n>`：引用命令历史中第 `n` 行命令（正序）
	-	`!-<n>`：引用当前命令前 `n` 行命令（逆序）
	-	`!<key>`：引用最近以 `key` 开头命令
		-	`key` 只匹配命令，不匹配参数
	-	`!?<key>`：引用最近包含 `key` 命令

-	历史记录操作：不关系命令内容，仅按照字符分隔确定
	-	`!$`：引用上调命令最后一个参数
	-	`!*`：引用上条命令所有参数
	-	`!:-`：上条命令除最后参数外全部（包括命令）
	-	`^<from>^<to>`：将最近包含 `from` 命令替换为 `to` 并执行

> - 此 `!` 被称为事件提示符，用于引用历史命令

-	`!cmd`：引用最近以`cmd`**开头**的命令，包括参数
	-	`!cmd:gs/pattern/replace`：将上条`cmd`开头命令中
		`pattern`替换为`replace`

###	`history` 及相关配置

-	`history`：显示操作历史
	-	参数
		-	缺省：带行号输出 `HISFILE` 指定文件的内容
		-	`-c`：清除历史记录

-	相关配置
	-	`HISFILE`：历史命令文件，一般为 `~/.bash_history`
	-	`HISTTIMEFORMAT`：指定历史操作时间戳格式
	-	`HISSIZE`：历史操作保存数量
	-	`HISTIGNORE`：指定不写入记录的命令（`:` 分隔）

> - *Bash* 独有

##	工作目录

###	`cd`

-	`cd`：进入指定目录
	-	`$ cd <path>`

-	参数
	-	缺省：`$HOME` 用户主目录
	-	`-`：上次目录

-	相关变量
	-	`$CDPATH`：`cd` 的搜索路径，`:` 分隔多个路径

###	`dirs`、`pushd`、`popd`

-	`dirs`：列出目录堆栈
	-	说明
		-	目录堆栈栈顶总是当前目录，**改变目录栈顶必然切换至栈顶目录**
	-	参数
		-	`-c`：清空目录堆栈（仅保留
		-	`-l`：将用户主目录替换为完整路径
		-	`-p`：栈顶至栈底，每行打印一个目录项
		-	`-v`：带编号每行打印
		-	`+<num>`：列出从栈顶开始第 `num` 号目录
		-	`-<num>`：列出从栈底开始第 `num` 号目录

-	`pushd`：将目录放入目录堆栈
	-	参数
		-	缺省：交换目录堆栈栈顶两个目录
		-	`<path>`：将 `path` 压入栈顶
		-	`+<num>`：将栈顶开始第 `num` 目录移至栈顶
		-	`-<num>`：将栈底开始第 `num` 目录移至栈顶

-	`popd`：从目录堆栈中移除目录
	-	参数
		-	缺省：移除目录堆栈顶层目录
		-	`-n`：删除目录堆栈栈顶下个记录
		-	`+<num>`：移除栈顶开始第 `num` 目录
		-	`-<num>`：移除栈底开始第 `num` 目录

> - *Bash* 独有

##	命令执行

-	`<space>`/`\t`：参数分隔符
	-	在不同参数直接间区分不同参数
	-	多个空白符会被自动忽略
-	`;`：命令结束符
	-	行内分割多条命令（换行时无需），允许一行内放置多条命令
	-	`;` 后命令总会执行，无论前序命令执行结果

###	条件命令

-	可以利用命令间的逻辑运算、短路求值特性简化条件命令执行
	-	`&&`：前序命令执行成功才会继续执行后序命令
	-	`||`：前序命令执行失败 **才会** 继续执行后序命令
	-	`<cmd1> && <cmd2> || <cmd3>`：`cmd1` 执行成功则执行 `cmd2`，否则执行 `cmd3`

###	`alias`

-	`alias`：为命令指定别名
	-	参数：`alias <ALIAS>=<COMMAND-DEF>`
-	`unalias`：解除别名

###	命令组

-	命令组：重定向将被应用于组内全部命令
	-	`(<cmd-list>)`：创建子 Shell 执行组内命令
		-	`()` 是操作符，被认为是分隔标记，无需用空格和内部命令分隔
	-	`{ <cmd-list> }`：在当前 Shell 上下文中执行组内命令
		-	`{}` 是保留字，与其中命令必须用空白分隔

###	`-` 连词

-	`--`：命令行中转义 `-`，避免 Shell 将 `-` 解释为参数选项

	```shell
	$ rm -- -file
		# 删除文件`-file`
	```

##	帮助、信息

###	`type`

-	`type`：查看命令定义、来源
	-	`$ type <op> [<name>]`：找到任意定义时返回 0，否则返回非 0

-	选项参数
	-	`-a`：查看 `name` 的所有定义：内建、可执行文件、别名、函数
	-	`-t`：打印单个单词表示命令类型
		-	未查找到则不打印，并返回非 0 值
	-	`-p`：打印可执行文件的路径或空，若 `-t` 选项将返回 `file`
	-	`-P`：即使 `-t` 选项不反悔 `file`，也强制查找执行文件路径
	-	`-f`：忽略函数

###	`help`

-	`help`：Bash 内建、关键字帮助
	-	`$ help`：缺省列出所有 Shell 内建、关键字
	-	`$ help <op> <ptn>`：打印匹配 `ptn` 的帮助信息

-	选项参数
	-	`-d`：打印说明信息
	-	`-m`：以 pseudo-manpage 格式打印信息
	-	`-s`：打印用法缩略


