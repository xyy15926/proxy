---
title: Pywin32 图形化模块
categories:
  - Python
  - Pywin32
tags:
  - Python
  - GUI
date: 2021-05-31 11:10:44
updated: 2021-05-31 11:10:44
toc: true
mathjax: true
description: 
---

#	`win32gui`

> - `win32gui` 提供对原生 *win32 GUI API* 的接口

##	*Device Context*

-	`win32gui.CreateDC(Driver, Device, InitData)`：为打印机、显示器创建设备上下文
	-	`Driver`：显示器、打印机供应商名称
	-	`Device`：设备名，如打印机名称
	-	`InitData`：`PyDEVMODE` 指定打印参数
-	`win32gui.CreateCompatibleDC(hdc)`：创建兼容指定设备的内存设备上下文
	-	`hdc`：上下文句柄
-	`win32gui.SetMapMode(hdc, MapMode)`：设置逻辑（页空间）单位至设备（空间）单位的映射方式
	-	`MapMode`：映射方式，可通过 `win32con.MM_<XXXX>` 查看枚举值
-	`win32gui.SetViewportExtEx(hdc,XExtent,YExtent)`：设置 *DC* 可视区域的 *extent*
	-	`XExtent`/`YExtent`：`X`/`Y` 方向逻辑长度
-	`win32gui.SetWindowExtEx(hdc,XExtent,YExtent)`：设置 *DC* 窗口的长度
-	`win32gui.SetWindowOrg(hdc,X,Y)`：设置 *DC* 可视区域起始位置
	-	`X`/`Y`：`X`/`Y` 方向的逻辑坐标

> - 设置逻辑长度即将 *DC* 的可视区域空间进行线性变换（映射），而避免对需展示信息进行线性变换
> - *viewport* 是整个可视区域，*window* 是可是区域中用于展示信息的部分，*windowOrg* 则是 *window* 在信息中的偏移

##	窗口

-	`win32gui.EnumWindows(callback, ctx)`：遍历所有窗口

	```python
	def winEnumHandler(hwnd, ctx):
		if win32gui.IsWindow(hwnd) and win32gui.IsWindowsEnabled(hwnd) and wind32gui.IsWindowVisible(hwnd):
			print(hwnd, win32gui.GetWindowText(hwnd))
			ctx.append(hwnd)
	hwnd_list = []
	win32gui.EnumWindows(winEnumHandler, hwnd_list)
	```

-	`win32gui.EnumChildWindows(parent_hwnd, callback, ctx)`：遍历窗口的子窗口

-	`win32gui.ShowWindow(hwnd, status)`：根据`status`设置窗口状态

	|常量|大小|显示状态|激活状态|
	|-----|-----|-----|-----|
	|`win32con.SW_HIDE`|不变|隐藏|不变|
	|`win32con.SW_MAXSIZE`|最大化|不变|不变|
	|`win32con.SW_MINISIZE`|最小化|不变|不变|
	|`win32con.SW_RESTORE`|恢复正常|不变|不变|
	|`win32con.SW_SHOW`|不变|显示|激活|
	|`win32con.SW_SHOWMAXIMIZED`|最大化|显示|激活|
	|`win32con.SW_SHOWMINIMIZED`|最小化|显示|激活|
	|`win32con.SW_SHOWNORMAL`|恢复正常|显示|激活|
	|`win32con.SW_SHOWMINNOACTIVE`|最小化|显示|不变|
	|`win32con.SW_SHOWNA`|不变|显示|不变|
	|`win32con.SW_SHOWNOACTIVE`|恢复正常|显示|不变|

-	`win32gui.SetForegroundWindow(hwnd)`：

#	`win32ui`

-	`win32ui`、`win32gui` 差别
	-	`win32gui` 是对微软底层接口的直接封装
	-	`win32ui` 则是对 `win32gui` 

##	*Device Context*

-	`win32ui.CreateDC()`：创建未初始化的设备上下文
-	`win32ui.CreateDCFromHandle(handle)`：从句柄中创建上下文

> - <https://newcenturycomputers.net/projects/pythonicwindowsprinting.html>

