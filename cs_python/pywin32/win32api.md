---
title: Pywin32 接口
categories:
  - Python
  - Pywin32
tags:
  - Python
  - Pywin32
  - Shell
  - Automatic
date: 2021-05-19 10:46:48
updated: 2021-05-19 10:46:48
toc: true
mathjax: true
description: 
---

#	说明

> - <http://timgolden.me.uk/pywin32-docs/win32_modules.html>

-	*pywin32* 包是对 *windows* 提供接口的封装，和 *C/CPP* 接口内容相近

#	`win32api`

##	Shell

-	`win32api.ShellExecute(hwnd, op, file, params, dir, bShow)`
	-	`hwnd`：父窗口句柄
	-	`op`：操作，`"open"`、`"print"`
	-	`file`：需要执行操作的文件
	-	`params`：待执行操作的文件为可执行文件时，需要传递的参数
	-	`dir`：应用执行的初始文件夹
	-	`bShow`：应用打开时是否展示

##	Keyboard

-	`win32api.keybd_event(bVk, bScan, dwFlags, dwExtraInfo) )`
	-	`bVk`：虚拟键码，可通过 `win32con.VK_<KEY>` 获取
	-	`bScan`：硬件扫描码，一般设置为 `0`
	-	`dwFlags`：一般设置为 `0`，表示按键按下
		-	`win32con.KEYEVENTF_EXTENDEDKEY`
		-	`win32con.KEYEVENTF_KEYUP`：按键释放
	-	`dwExtraInfo`：与击键相关的附加的32位值，一般设置为 `0`





