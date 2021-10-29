---
title: Bash 内建关键字、命令
categories:
  - Linux
  - Shell
tags:
  - Linux
  - Bash
  - Shell
  - Builtin
date: 2021-08-10 16:29:08
updated: 2021-10-29 11:20:24
toc: true
mathjax: true
description: 
---

##	Bash 内置关键字

###	Bash 脚本执行关键字

-	执行
	-	`trap [-lp] [[arg] signal_spec ...]`
	-	`exit [n]`
	-	`ulimit [-SHabcdefiklmnpqrstuvxPT] [limit]`
	-	`time [-p] pipeline`：统计命令耗时
	-	`times`：显示进程累计时间

-	参数相关
	-	`shift [n]`
	-	`getopts optstring name [arg]`

-	输入、输出
	-	`printf [-v var] format [arguments]`
	-	`echo [-neE] [arg ...]`
	-	`readarray [-d delim] [-n count] [-O origin] [-s count] [-t] [-u fd] [-C callback] [-c quant>`
	-	`mapfile [-d delim] [-n count] [-O origin] [-s count] [-t] [-u fd] [-C callback] [-c quantum>`
	-	`read [-ers] [-a array] [-d delim] [-i text] [-n nchars] [-N nchars] [-p prompt] [-t timeout>`

###	脚本执行流

-	分支、循环
	-	`if COMMANDS; then COMMANDS; [ elif COMMANDS; then COMMANDS; ]... [ else COMMANDS; ] fi`
	-	`while COMMANDS; do COMMANDS; done`
	-	`until COMMANDS; do COMMANDS; done`
	-	`case WORD in [PATTERN [| PATTERN]...) COMMANDS ;;]... esac`
	-	`select NAME [in WORDS ... ;] do COMMANDS; done`
	-	`continue [n]`
	-	`break [n]`
	-	`for NAME [in WORDS ... ] ; do COMMANDS; done`
	-	`for (( exp1; exp2; exp3 )); do COMMANDS; done`
	-	`:`
	-	`{ COMMANDS ; }`

-	条件检查
	-	`test [expr]`
	-	`[ arg... ]`
	-	`[[ expression ]]`
	-	`true`
	-	`false`

-	变量、表达式
	-	`declare [-aAfFgilnrtux] [-p] [name[=value] ...]`
	-	`typeset [-aAfFgilnrtux] [-p] name[=value] ...`
	-	`unset [-f] [-v] [-n] [name ...]`
	-	`let arg [arg ...]`
	-	`(( expression ))`
	-	`local [option] name[=value] ...`
	-	`readonly [-aAf] [name[=value] ...] or readonly -p`
	-	`variables - Names and meanings of some shell variables`

-	函数
	-	`function name { COMMANDS ; } or name () { COMMANDS ; }`
	-	`return [n]`

###	环境设置

-	Shell、Bash 配置
	-	`set [-abefhkmnptuvxBCHP] [-o option-name] [--] [arg ...]`
	-	`shopt [-pqsu] [-o] [optname ...]`
	-	`export [-fn] [name[=value] ...] or export -p`

-	Shell 信息
	-	`help [-dms] [pattern ...]`
	-	`type [-afptP] name [name ...]`
	-	`logout [n]`

-	文件系统操作
	-	`pwd [-LP]`
	-	`pushd [-n] [+N | -N | dir]`：添加目录到目录堆栈顶部
	-	`popd [-n] [+N | -N]`
	-	`dirs [-clpv] [+N] [-N]`
	-	`cd [-L|[-P [-e]] [-@]] [dir]`
	-	`umask [-p] [-S] [mode]`

###	命令信息

-	历史命令
	-	`history [-c] [-d offset] [n] or history -anrw [filename] or history -ps arg [arg...]`
	-	`fc [-e ename] [-lnr] [first] [last] or fc -s [pat=rep] [command]`
	-	`hash [-lr] [-p pathname] [-dt] [name ...]`
	-	`read [-ers] [-a array] [-d delim] [-i text] [-n nchars] [-N nchars] [-p prompt] [-t timeout>`

-	别名、快捷键
	-	`alias [-p] [name[=value] ... ]`
	-	`unalias [-a] name [name ...]`
	-	`bind [-lpsvPSVX] [-m keymap] [-f filename] [-q name] [-u name] [-r keyseq] [-x keyseq:shell->`

-	命令补全
	-	`compgen [-abcdefgjksuv] [-o option] [-A action] [-G globpat] [-W wordlist]  [-F function] [->`
	-	`complete [-abcdefgjksuv] [-pr] [-DEI] [-o option] [-A action] [-G globpat] [-W wordlist]  [->`
	-	`compopt [-o|+o option] [-DEI] [name ...]`

###	任务执行

-	执行任务
	-	`builtin [shell-builtin [arg ...]]`：执行 bash 内建命令
	-	`source filename [arguments]`
	-	`. filename [arguments]`
	-	`eval [arg ...]`
	-	`exec [-cl] [-a name] [command [arguments ...]] [redirection ...]`
	-	`command [-pVv] command [arg ...]`

-	查看、控制任务
	-	`enable [-a] [-dnps] [-f filename] [name ...]`
	-	`jobs [-lnprs] [jobspec ...] or jobs -x command [args]`
	-	`job_spec [&]`
	-	`kill [-s sigspec | -n signum | -sigspec] pid | jobspec ... or kill -l [sigspec]`
	-	`fg [job_spec]`
	-	`bg [job_spec ...]`
	-	`suspend [-f]`
	-	`disown [-h] [-ar] [jobspec ... | pid ...]`
	-	`wait [-fn] [id ...]`
	-	`caller [expr]`
	-	`coproc [NAME] command [redirections]`



