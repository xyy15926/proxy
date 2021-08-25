---
title: Bash 脚本执行
categories:
  - Linux
  - Bash
tags:
  - Linux
  - Bash
  - Shell
  - Shebang
  - Execution
date: 2019-07-31 23:59:23
updated: 2021-08-25 23:06:58
toc: true
mathjax: true
comments: true
description: Shell脚本执行
---

##	脚本执行

###	`#!`

-	*Shebang*/*Hashbang*：`#!` 字符序列
	-	脚本首行的前两个字符，因此首行也被称为 *Shebang 行*
	-	其后指定指定脚本解释器
		-	`#!` 和脚本解释器之间空格为可选
		-	大部分脚本均可使用 *Shabang* 指定解释器
	-	*Shebang 行* 不必须
		-	包含 *Shebang 行* 时，可直接将脚本作为命令执行
		-	缺少时，则需要在命令行指定解释器

-	实务中指定脚本解释器
	-	直接写明解释器路径，但依赖对解释器的位置的了解
	-	使用 `env` 命令获取解释器位置
		-	`env` 总是位于 `/usr/bin/env`
		-	减少对机器配置依赖

	```shell
	#!/usr/bin/env python
	#!/usr/bin/env shell
	```

###	脚本执行方式

-	`$ <file-name>`：开启脚本解释器进程，在新进程中执行脚本
	-	说明
		-	需有文件的执行权限
		-	不影响当前 Shell 环境
		-	缺省只寻找 `PATH` 路径下文件，忽略当前目录
	-	`$ <sh> <file-name>`：同直接执行脚本，但自行指定脚本解释器

-	`$ source <file-name>`：读取文件内容至 Shell 中，然后在 Shell 里执行文件内容
	-	说明
		-	需有文件读权限
		-	可直接访问、影响当前 Shell 环境
	-	可用于
		-	加载外部脚本库，之后可以直接使用其中定义变量、函数
		-	直接利用当前 Shell 环境执行脚本
	-	`$ . <file-name>`：同 `source`，`source` 的简写

-	以如下脚本为例

	```shell
	#!/bin/bash
	echo "fisrt"
	sleep 3
	echo "second"
	```

	-	`$ test.sh` / `$ sh test.sh`：产生两个新进程 `test.sh` 和 `sleep`
		-	在 `second` 输出之前终止，会同时终止两个进程

	-	`$ source test.sh` / `$ . test.sh`：产生一个新进程 `sleep`
		-	在 `second` 输出之前终止，只有 `sleep` 进程被终止，剩余内容继续执行

###	`exit`

-	`exit`：终止脚本执行，并返回退出值
	-	`$ exit <N>`：`N` 为脚本的退出值

-	参数（约定，但建议遵守）
	-	`0`：成功返回
	-	`1`：发生错误
	-	`2`：用法错误
	-	`126`：非可执行脚本
	-	`127`：命令未找到
	-	若脚本被信号 `N` 终止，返回 `128+N`

###	`trap`

-	`trap`：响应系统信号，在接受到系统信号时执行指定命令
	`$ trap <do-cmd> <signal> [<signal>...]`

-	选项参数
	-	`-l`：列出所有系统信号

###	`ulimit`

-	`ulimit`：控制 Shell 启动进程的可获取资源
	-	`$ ulimit [-HS] <op> <limit>`

-	参数选项
	-	`-S`：改变、报告资源的软限制
	-	`-H`：改变、报告资源的硬限制
	-	`-a`：报告当前所有限制
	-	`-b`：socket 缓冲上限
	-	`-c`：核心文件上限
	-	`-d`：进程数据区上限
	-	`-e`：调度优先级上限
	-	`-f`：Shell 及其子进程写文件上限
	-	`-i`：pending 信号上限
	-	`-k`：kqueue 分配上限
	-	`-l`：内存锁定上限
	-	`-m`：保留集上限（很多
	-	`-n`：打开的文件描述符上限
	-	`-p`：管道缓冲上限
	-	`-q`：*POSIX* 信息队列容量上限
	-	`-r`：实时调度优先级上限
	-	`-s`：堆栈上限
	-	`-t`：每秒 CPU 时间上限
	-	`-u`：用户可用进程数量上限
	-	`-v`：Shell 可用虚拟内存上限
	-	`-x`：文件锁上限
	-	`-P`：虚拟终端上限
	-	`-R`：进程阻塞前可运行时间上限
	-	`-T`：线程数上限

###	`time`

-	`time`：测量指令执行耗时、资源
	-	`time` 是 Bash 的保留字，这允许 `time` 方便地测量内建、Shell 函数、管道
	-	`$TIMEFORMAT` 变量指定打印的格式

-	选项参数
	-	`-p`：按照可移植 *POSIX* 格式打印

##	脚本参数

###	`shift`

-	`shift`：左移参数列表，会修改参数列表
	-	说明
		-	可用于参数数量未知时遍历参数
	-	参数
		-	`<N>`：左移参数数量，缺省 1

###	`getopts`

-	`getopts`：取出脚本、函数所有配置项（即带有前置连词线 `-` 的单字符参数）
	-	用于结合 `while` 循环遍历所有参数、参数值
	-	遇到不带连词线参数时，执行失败，退出循环

-	参数：`<OPTSTR> <NAME>`
	-	`OPTSTR`：可配置参数名的汇总字符串
		-	参数字符后跟 `:`：配置项须带参数值
		-	参数字符后跟 `::`：配置项参数值可选，设置值必须紧贴参数字符
		-	字符串中首个 `:`：忽略错误
	-	`NAME`：当前取到的参数名称，即循环中临时变量
		-	参数在 `OPTSTR` 中未配置时，赋 `?`
		-	`OPTARG`：循环中存储带参数值参数的取值
		-	`OPTIDX`：原始 `$*` 中下个待处理参数位置，即已处理参数数量加 1（**包括获取的 `OPTARG`**）

```shell
function func (){
	echo OPTIND: $OPTIND
	while getopts ":a:B:cdef" opt; do
	  case $opt in
		a) echo "this is -a the arg is ! $OPTARG at $OPTIND" ;;
		B) echo "this is -B the arg is ! $OPTARG at $OPTIND" ;;
		c) echo "this is -c the arg is ! $OPTARG at $OPTIND" ;;
		\?) echo "Invalid option: -$OPTARG" ;;
	  esac
	done
	echo OPTIND: $OPTIND
	echo $@
	shift $(($OPTIND - 1))
	echo $@

}
func -a 23 -B 1904-03-04 343 age
```

##	输入输出

###	`echo`

-	`echo`：向标准输出输出其参数
	-	单行文本可以省略引号，多行文本需用引号扩起
	-	参数
		-	`-n`：取消末尾回车符
		-	`-e`：对引号内字符转义

	> - 存在 `echo` 同名程序

###	`read`

```shell
read -p "Enter names: " name

while read line:
do
	echo $line
done <<< $string
```

-	`read`：从标准输入中读取单词，赋值给参数
	-	`$ read <op> <var1>[ <var2>...]`
	-	读取用户输入直到遇到 `-d` 指定的结束符

-	选项参数
	-	`-t`：超时秒数
	-	`-p <MSG>`：提示信息
	-	`-a <ARR>`：将用户输入存储在数组 `ARR` 中
	-	`-n <NUM>`：读取 `NUM` 数量字符，或遇到分隔符后停止读取
	-	`-N <NUM>`：读取 `NUM` 数量字符停止读取
	-	`-e`：允许用户输入是使用 *readline* 快捷键
	-	`-d <DELIMITER>`：以 `DELIMITER` 的首个字符作为用户输入结束，缺省为 `\n`
	-	`-r`：*raw* 模式，不将反斜杠作为转义符号
	-	`-u <FD>`：从文件描述符 `FD` 而不是标准输入中读取
	-	`-s`：隐藏输入

-	参数
	-	`var1`：存储用户输入的变量名，若未定义，则缺省为 `REPLY`
		-	若用户输入项（单词）少于 `read` 参数中变量数目，则额外变量值为空
		-	若用户输入项（单词）多于 `read` 参数中变量数目，则多余输入项存储在最后变量中

###	`readarray`

-	`readarray`/`mapfile`：从标准输入中读取行，作为数组成员
	-	读取用户输入直到遇到输入终止符 `EOF`（`C-d` 终止）

-	选项参数
	-	`-t`：删除输入行文本末尾换行符
	-	`-n <NUM>`：最多读取 `NUM` 行，缺省为 0，读取所有输入
	-	`-o <ORI>`：数组赋值起始位置
	-	`-s <START>`：忽略起始的 `START` 行
	-	`-u <FD>`：从文件描述符 `FD` 而不是标准输入中读取
	-	`-c <COUNT>`：缺省为 5000
	-	`-C <CALLBACK>`：读取 `COUNT` 行之后执行 `CALLBACK`

###	`--`

-	`--`：配置项终止符，指示气配置项结束，之后 `-` 当作实体参数解释

###	*Pipeline*

-	管道：用 `|`、`|&` 分隔命令序列
	-	前个命令的输出通过管道连接到后个命令输入
		-	优先级高于命令指定的重定向
	-	`|&` 是 `2>&1 |` 的简写
		-	命令的错误输出、标准输出均被连接
		-	错误输出的隐式重定向优先级低于任何命令指定的重定向

-	管道中命令的执行
	-	若命令异步执行，Shell 将等待整个管道中命令执行完毕
	-	管道中命令在独立的 Shell 中执行
		-	若 `shopt` 的 `lastpipe` 选项被设置，管道中最后命令可能在当前 Shell 中执行

-	管道有其整体退出状态（管道前加上 `!` 表达对管道整体退出状态取反）
	-	若 `set` 的 `pipefail` 选项为被启用，退出状态由最后命令决定
	-	否则，全部成功才返回 0，否则返回最后非 0 退出状态

###	*Redirection*

-	重定向：在命令执行前，可以使用特殊符号重定向其输入、输出
	-	文件句柄的复制、打开、关闭、指向其他文件
	-	改变命令的读取、写入目标

-	重定向存续超过命令执行
	-	而，重定向符号前用文件描述符（数字）而不是文件名指定目标
	-	故，可以手动管理文件描述符的生命周期（而不依赖于命令）

####	输入、输出重定向

-	`<FD> < <WORD>`：输入重定向
	-	在文件描述符 `FD` 上以读取的方式打开文件 `WORD`
		-	`FD` 缺省为 0，即标准输入

-	`<FD> >[|] <WORD>`：输出重定向
	-	在文件描述符 `FD` 上以写入的方式打开文件 `WORD`
		-	`FD` 缺省为 1，即标准输出
	-	若文件 `WORD` 不存在则被创建，若文件存在
		-	重定向符为 `>|` 、或 `set` 的 `noclobber` 未被设置，文件被截断为空再写入
		-	若 `set` 的 `noclobber` 选项被设置且 `>`，文件 `WORD` 存在则被重定向失败
	-	`>` 替换为 `>>` 即为 *appending* 重定向

-	`<FD> <> <WORD>`：输入、输出重定向
	-	在文件描述符 `FD` 上以读取、写入的方式打开文件 `WORD`
		-	`FD` 缺省为 0，即标准输入
	-	若文件 `WORD` 不存在则被创建

-	`&> <WORD>`、`>& <WORD>`：标准输出、错误重定向至文件 `WORD`
	-	二者语义上等同于 `> <WORD> 2>&1`
		-	但 `WORD` 可能被扩展为数字、`-`，导致 `>& <WORD>` 被解释为复制
	-	`>` 替换为 `>>` 即为 *appending* 重定向

> - 以上 `WORD` 支持扩展
> - 重定向符首个字符为 `<`，缺省目标为 `0`；为 `>`，缺省目标为 `1`

####	文件描述符复制、移动、关闭

-	`<FD> <& <FD-INPUT>`：复制输入文件描述符（可读取）`FD-INPUT` 至 `FD`
	-	`FD-INPUT` 应扩展为有效的输入文件描述符，否则发生重定向错误
		-	`FD-INPUT` 扩展为 `-` 时，`FD` 被关闭

-	`<FD> >& <FD-OUTPUT>`：复制输出文件描述符（可写入）`FD-OUTPUT` 至 `FD`
	-	`FD-OUTPUT` 扩展为非数字、`-`时，将被解释为标准输出、错误重定向
		-	`FD-INPUT` 扩展为 `-` 时，`FD` 被关闭

-	在被复制的文件描述符之后添加 `-`，则会将被复制描述符关闭
	-	`<FD> <& <FD-INPUT>-`：输入文件描述符移动
	-	`<FD> >& <FD-OUTPUT>-`：输出文件描述符移动

> - `&` 复制是必要的，否则指示单纯的多次重定向，并覆盖之前

####	*Here Document*

```shell
<FD> << <TOKEN>
	# here doc content
<DELIMITER>
```

-	*Here* 文档：将 `TOKEN`、`DELIMITER` 之间的字符串重定向至 `FD`
	-	`TOKEN` 不执行变量扩展、命令替换、算术扩展、文件名扩展
	-	若 `TOKEN` 包含标记
		-	`DELIMITER` 为 `TOKEN` 执行标记移除的结果
		-	*here document* 不执行扩展
	-	若 `TOKEN` 不包含标记
		-	*here document* 执行变量扩展、命令替换、算术扩展（相应引导符需要被转义）
		-	`\n` 被忽略

-	`<<-` 作为重定向符时
	-	前导的制表符被剔除，包括包含 `DILIMITER` 行

####	*Here String*

-	*Here* 字符串：`<<< <WORD>`
	-	将 `WORD` 重定向至 `FD`
	-	`WORD` 执行 *Tilde* 扩展、变量扩展、命令替换、算术扩展、标记移除
		-	结果作为字符串，末尾添加换行符

####	特殊文件

-	`/dev/fd/<fd>`：文件描述符 `fd` 被复制
-	`/dev/stdin`：文件描述符 `0`（标准输入）被复制
-	`/dev/stdout`：文件描述符 `1`（标准输出）被复制
-	`/dev/stderr`：文件描述符 `2`（标准错误）被复制
-	`/dev/tcp/host/port`：尝试打开相应的 *TCP* socket
-	`/dev/udp/host/port`：尝试打开相应的 *UDP* socket


