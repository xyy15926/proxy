---
title: Linux Interrupt
categories:
  - Linux
  - IPC
tags:
  - Linux
  - Kernel
  - IPC
  - Interrupt
date: 2020-09-21 09:06:45
updated: 2020-09-21 09:06:45
toc: true
mathjax: true
description: Linux Interrupt
---

##	中断

-	软中断：

-	硬中断：外围设备完成用户请求后，会向CPU发出中断信号
	-	CPU会暂停执行下条将要执行的指令，转而执行中断信号
		对应处理程序，并将进程转入内核态

##	Interrupt

-	中断属性
	-	中断号：标识不同中断
	-	中断处理程序：不同中断有不同处理程序

-	*interrupt vector table*：内核中维护，存储所有中断处理
	程序地址
	-	中断号就是相应中断在中断向量表中偏移量









