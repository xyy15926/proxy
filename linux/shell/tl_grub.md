---
title: Grub
categories:
  - Linux
  - Tool
tags:
  - Linux
  - Tool
  - Grub
  - Boot Loader
date: 2021-07-29 21:36:22
updated: 2021-10-26 19:42:59
toc: true
mathjax: true
description: 
---

##	*GRUB*

-	*GRUB*：来自 *GNU* 项目的启动引导程序
	-	是多启动规范的实现，允许用户在主机内同时拥有多个操作系统，并在开机时选择期望运行的系统
	-	特点
		-	拥有丰富终端命令，可动态配置启动项：在启动时加载配置，并允许在启动时修改
		-	支持多种可执行格式、链式启动、无盘系统启动
		-	轻便、用户界面丰富

##	*GRUB* 配置文件

-	`/etc/sysconfig/grub`：*GRUB* 配置文件，实际上是 `/etc/default/grub` 的软连接
