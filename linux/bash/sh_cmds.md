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
updated: 2021-08-25 23:09:34
toc: true
mathjax: true
description: 
---

##	Bash 内建关键字

###	任务相关

-	`builtin [shell-builtin [arg ...]]`：执行 bash 内建命令
-	`enable [-a] [-dnps] [-f filename] [name ...]`
-	`jobs [-lnprs] [jobspec ...] or jobs -x command [args]`
-	`job_spec [&]`
-	`kill [-s sigspec | -n signum | -sigspec] pid | jobspec ... or kill -l [sigspec]`
-	`fg [job_spec]`
-	`bg [job_spec ...]`
-	`suspend [-f]`
-	`disown [-h] [-ar] [jobspec ... | pid ...]`
-	`source filename [arguments]`
-	`. filename [arguments]`
-	`command [-pVv] command [arg ...]`
-	`eval [arg ...]`
-	`exec [-cl] [-a name] [command [arguments ...]] [redirection ...]`
-	`wait [-fn] [id ...]`
-	`type [-afptP] name [name ...]`
-	`ulimit [-SHabcdefiklmnpqrstuvxPT] [limit]`
-	`logout [n]`

###	环境相关

-	`set [-abefhkmnptuvxBCHP] [-o option-name] [--] [arg ...]`
-	`unset [-f] [-v] [-n] [name ...]`
-	`export [-fn] [name[=value] ...] or export -p`
-	`history [-c] [-d offset] [n] or history -anrw [filename] or history -ps arg [arg...]`
-	`fc [-e ename] [-lnr] [first] [last] or fc -s [pat=rep] [command]`
-	`alias [-p] [name[=value] ... ]`
-	`unalias [-a] name [name ...]`
-	`pwd [-LP]`
-	`bind [-lpsvPSVX] [-m keymap] [-f filename] [-q name] [-u name] [-r keyseq] [-x keyseq:shell->`
-	`pushd [-n] [+N | -N | dir]`：添加目录到目录堆栈顶部
-	`popd [-n] [+N | -N]`
-	`dirs [-clpv] [+N] [-N]`
-	`cd [-L|[-P [-e]] [-@]] [dir]`
-	`umask [-p] [-S] [mode]`
-	`printf [-v var] format [arguments]`
-	`echo [-neE] [arg ...]`

###	执行流控制

-	`(( expression ))`
-	`if COMMANDS; then COMMANDS; [ elif COMMANDS; then COMMANDS; ]... [ else COMMANDS; ] fi`
-	`case WORD in [PATTERN [| PATTERN]...) COMMANDS ;;]... esac`
-	`:`
-	`[ arg... ]`
-	`let arg [arg ...]`
-	`[[ expression ]]`
-	`local [option] name[=value] ...`
-	`break [n]`
-	`getopts optstring name [arg]`
-	`read [-ers] [-a array] [-d delim] [-i text] [-n nchars] [-N nchars] [-p prompt] [-t timeout>`
-	`readarray [-d delim] [-n count] [-O origin] [-s count] [-t] [-u fd] [-C callback] [-c quant>`
-	`mapfile [-d delim] [-n count] [-O origin] [-s count] [-t] [-u fd] [-C callback] [-c quantum>`
-	`continue [n]`
-	`shift [n]`
-	`for NAME [in WORDS ... ] ; do COMMANDS; done`
-	`for (( exp1; exp2; exp3 )); do COMMANDS; done`
-	`test [expr]`
-	`while COMMANDS; do COMMANDS; done`
-	`{ COMMANDS ; }`
-	`declare [-aAfFgilnrtux] [-p] [name[=value] ...]`
-	`readonly [-aAf] [name[=value] ...] or readonly -p`
-	`select NAME [in WORDS ... ;] do COMMANDS; done`
-	`return [n]`
-	`until COMMANDS; do COMMANDS; done`
-	`function name { COMMANDS ; } or name () { COMMANDS ; }`

####	布尔标志

-	`true`
-	`false`

###	工具

-	`hash [-lr] [-p pathname] [-dt] [name ...]`
-	`time [-p] pipeline`：统计命令耗时
-	`times`：显示进程累计时间

###	其他1

-	`caller [expr]`
-	`compgen [-abcdefgjksuv] [-o option] [-A action] [-G globpat] [-W wordlist]  [-F function] [->`
-	`complete [-abcdefgjksuv] [-pr] [-DEI] [-o option] [-A action] [-G globpat] [-W wordlist]  [->`
-	`compopt [-o|+o option] [-DEI] [name ...]`
-	`coproc [NAME] command [redirections]`
-	`shopt [-pqsu] [-o] [optname ...]`
-	`typeset [-aAfFgilnrtux] [-p] name[=value] ...`
-	`variables - Names and meanings of some shell variables`

###	其他2

-	`trap [-lp] [[arg] signal_spec ...]`
-	`exit [n]`
-	`help [-dms] [pattern ...]`

