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
updated: 2021-08-09 20:43:50
toc: true
mathjax: true
description: 
---

##	任务管理

##	命令总结

-	系统监控
	-	`scar`：收集、报告、保存系统活动信息
	-	`iostat`：报告 CPU 统计数据，设备、分区输入/输出信息
	-	`iotop`：*I/O* 监控
	-	`mpstat`：报告 CPU 相关统计数据
	-	`vmstat`：报告虚拟内存统计
	-	`tload`：加载显示系统平均负载、指定 tty 终端平均负载
	-	`time`：统计命令耗时 **（Bash 内建）**
	-	`uptime`：显示系统已运行时间
	-	`ipcs`：提供 IPC 设施信息 
	-	`ipcrm`：删除消息队列、信号量集、共享内存 ID
	-	`lslk`/`lslocks`：列出本地锁

-	进程查看
	-	`ps`：查看当前进程瞬时快照 
	-	`top`：显示当前正在运行进程（动态更新）
		-	按照**使用内存**大小排序，可以用于查找内存使用情况
	-	`pgrep`：按名称、属性查找进程
	-	`pidof`：根据进程名查找正在运行的进程进程号 
	-	`fuser`：显示使用指定文件、socket 的进程
	-	`pmap`：报告进程的内存映射
	-	`lsof`：列出打开的文件 <https://linuxtools-rst.readthedocs.io/zh_CN/latest/tool/lsof.html>

-	进程管理
	-	`kill`：终止进程
	-	`killall`：按名称终止进程
	-	`pkill`：按名称、属性终止进程
	-	`timeout`：在指定时间后仍然运行则终止进程
	-	`wait`：等待指定进程
	-	`chkconfig`：更新、查询、修改不同运行级别系统服务

-	Job 管理
	-	`&`：命令后台执行（放在命令之后）
	-	`nohup`：不挂起 job，即使 *shell* 退出
		-	`-p`：不挂起指定 *PID* 进程
	-	`jobs`：列出活动的作业
		-	`-l`：返回任务编号、进程号
	-	`bg`：将作业移至后台
		-	`%<no>`：将编号 `no` 号任务移至后台
	-	`fg`：将程序、命令放在前台执行
		-	`%<no>`：将编号 `no` 号任务移至前台
	-	`setsid`：在一个新的会话中运行程序
		-	等于在 `()` 中执行

		```sh
		$ setsid ./test.sh &`
			# 新会话中非中断执行程序，此时当前shell退出不会终止job
		$ (./test.sh &)
			# 同`setsid`，用`()`括起，进程在subshell中执行
		```

	-	`disown`：将进程从 jobs 列表中移除

		```shell
		$ disown -h %job_id
			# *放逐*已经在后台运行的job，
			# 则即使当前shell退出，job也不会结束
		```

	-	`screen`：创建断开模式的虚拟终端

		```bash
		$ screen -dmS screen_test
			# 创建断开（守护进程）模式的虚拟终端screen_test
		$ screen -list
			# 列出虚拟终端
		$ screen -r screen_test
			# 重新连接screen_test，此时执行的任何命令都能达到nohup
			```

##	快捷键

-	`<c-z>`：挂起当前任务
-	`<c-c>`：结束当前任务

