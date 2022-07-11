---
title: Python并行
categories:
  - Python
  - Py3std
tags:
  - Python
  - Py3std
  - Parallel
date: 2019-06-09 23:42:37
updated: 2019-06-09 23:42:37
toc: true
mathjax: true
comments: true
description: Python并行
---

##	*Global Intepretor Lock*

-	*GIL* 全局内存锁：任何 CPython 字节码执行前必须获得的解释器锁
	-	避免多个线程同时操作变量导致内存泄漏、错误释放
	-	在任何时刻，只能有一个线程处于工作状态
	-	线程在以下情况下会释放 *GIL*
		-	线程进入 *IO* 操作前
		-	解释器不间断运行一定时间

> - `sys.getswitchinterval()` 可查看解释器检查线程切换频率

-	*GIL* 特点
	-	单线程情况下，*GIL* 比给所有对象引用计数加锁性能更好、稳定
	-	*GIL* 实现简单，只需要管理一把解释器锁就能保证粗粒度线程安全
		-	但，*GIL* 只能保证引用计数正确，避免由此导致内存问题
		-	且，不能保证并发更新时的线程安全
	-	*GIL* 方便兼容 C 遗留库（要求线程安全的内存管理）
	-	*GIL* 导致 CPython 多线程对 *CPU* 密集型工作基本无提升

> - Python3.2 前因为切换锁的开销，多线程甚至会大幅降低 *CPU* 密集型工作效率
> - *GIL* 不是 Python 特性，是 CPython 引入的概念，JPython 中就没有 *GIL*

##	`threding`

##	`_thread`

##	`_dummy_thread`

##	`dummy_threading`

##	`multiprocessing`

##	`concurrent`

##	`concurrent.futures`

##	`subprocess`

##	`sched`

##	`queue`


