---
title: Linux 进程调度命令
categories:
  - Linux
  - Process Schedual
tags:
  - Linux
  - Kernel
  - Process Schedual
  - Process
date: 2021-07-29 21:50:04
updated: 2021-07-29 21:50:04
toc: true
mathjax: true
description: 
---

##	进程管理

###	`ps`

查看当前进程瞬时快照

###	`top`

显示当前正在运行进程（动态更新）

-	按照**使用内存**大小排序，可以用于查找内存使用情况

###	`pgrep`

按名称、属性查找进程

###	`pidof`

根据进程名查找正在运行的进程进程号

###	`kill`

终止进程

###	`killall`

按名称终止进程

###	`pkill`

按名称、属性终止进程

###	`timeout`

在指定时间后仍然运行则终止进程

###	`wait`

等待指定进程

###	`fuser`

显示使用指定文件、socket的进程

###	`pmap`

报告进程的内存映射

###	`lsof`

列出打开的文件

###	`chkconfig`

为系统服务更新、查询运行级别信息

##	作业

###	`&`

放在命令之后，命令后台执行

```shell
$ ./pso > pso.file 2>&1 &
	# 将`pso`放在后台运行，把终端输出（包括标准错误）
		# 重定向的到文件中
```

###	`nohup`

不挂起job，即使shell退出

```shell
$ nohup ./pso > pso.file 2>&1 &
	# 不挂起任务，输出重定向到文件
$ nohup -p PID
	# 不挂起某个进程
```

###	`jobs`

列出活动的作业

`-l`：返回任务编号、进程号

###	`bg`

恢复在后台暂停工作的作业

```shell
$ bg %n
	# 将编号为`n`的任务转后台运行
```

###	`fg`

将程序、命令放在前台执行

```shell
$ fg %n
	# 将编号为`n`的任务转前台运行
```

###	`setsid`

在一个新的会话中运行程序

```shell
$ setsid ./test.sh &`
	# 新会话中非中断执行程序，此时当前shell退出不会终止job
$ (./test.sh &)
	# 同`setsid`，用`()`括起，进程在subshell中执行
```

###	`disown

```shell
$ disown -h %job_id
	# *放逐*已经在后台运行的job，
	# 则即使当前shell退出，job也不会结束
```

###	`screen`

创建断开模式的虚拟终端

```bash
$ screen -dmS screen_test
	# 创建断开（守护进程）模式的虚拟终端screen_test
$ screen -list
	# 列出虚拟终端
$ screen -r screen_test
	# 重新连接screen_test，此时执行的任何命令都能达到nohup
	```

###	快捷键

-	`<c-z>`：挂起当前任务
-	`<c-c>`：结束当前任务

