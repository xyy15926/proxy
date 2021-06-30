---
title: Pywin32 设备
categories:
  - Python
  - Pywin32
tags:
  - Python
  - Device
  - Driver
  - Automatic
date: 2021-05-31 11:14:47
updated: 2021-05-31 11:14:47
toc: true
mathjax: true
description: 
---

#	`win32print`

> - <http://timgolden.me.uk/pywin32-docs/win32print.html>
> - <https://blog.csdn.net/sdfdsdfdf/article/details/106406291>

-	参数
	-	`level`：返回信息类型
		-	`1`：列表，无值含义说明
		-	`2-5`：内容逐渐省略的字典，键值对表示值含义、值
	-	`hPrinter`：`win32print.OpenPrinter` 返回的打印机句柄
	-	`hdc`：`win32gui.CreateDC` 返回打印机设备上下文句柄
	-	`pPrinter`：`win32print.GetPrinter` 返回的 `PRINTER_INFO_*` 字典

##	打印机

-	`win32print.OpenPrinter(printer, Defaults)`：获取指定打印机的句柄
-	`win32print.GetPrinter(hPrinter, Level)`：获取打印机相关信息
	-	`Level`：信息展示格式，一般设置为 `2` 以字典形式返回
-	`win32print.GetDevicesCaps(hdc, Index)`：获取设备上下文相关的参数、设置
	-	`Index` 参数代号，可以通过 `win32con.` 获取
	-	将打印机句柄作为 `hdc` 传递时，正常执行但返回结果始终为 `0`
-	`win32print.SetPrinter(hPrinter, Level, pPrinter, Command)`：改变打印机配置、状态
	-	`Command`：发送给打印机的命令，可以通过 `win32print.PRINTER_CONTROL_*` 查看
-	`win32print.EnumPrinters(flags, name, level)`：枚举打印机
	-	`flags`：指定打印机类型，可以通过 `win32print.PRINTER_ENUM_<type>` 获取相关类型枚举值
		-	可设置为 `6` 以枚举 *local*、*connections* 类型设备
-	`win32print.GetDefaultPrinter`/`win32print.GetDefaultPrinterW`
-	`win32print.SetDefaultPrinter`/`win32print.SetDefaultPrinterW`

###	打印机属性设置

-	打印机属性设置方式

	```python
	// 获取打印相关属性
	properties = win32print.GetPrinter(hPrinter, 2)
	devmode = properties["pDevMode"]
	// 修改属性设置
	properties["pDevMode"] = devmode
	win32print.SetPrinter(hPrinter, 2, properties, 0)
	```

-	打印机属性常用设置
	-	`devmode.Color`：彩印
		-	可通过 `win32con.DMCOLOR_<XXX>` 查看枚举值
	-	`devmode.PaperSize`：纸张尺寸
		-	可以通过 `win32con.DMPAPER_<XXXX>` 查看枚举值
		-	指定此属性时，`devmode.PaperLength`、`devmode.PaperWidth` 不生效
	-	`devmode.PaperLength`：纸张长，单位毫米
	-	`devmode.PaperWidth`：纸张宽，单位毫米
	-	`devmode.Duplex`：单双面（双面分为 *flip over* 、*flip up*）
		-	可通过 `win32con.DMDUP_<XXXX>` 查看枚举值
	-	`devmode.Orientation`：打印方向，即横向、纵向
		-	可通过 `win32con.DMORIENT_<XXXX>` 查看枚举值
		-	真实打印时影响不大，打印至 *pdf* 文件时、双面打印时有影响
	-	`devmode.MediaType`：纸张类型
	-	`devmode.DefaultSource`：纸张来源

> - <http://timgolden.me.uk/pywin32-docs/PyDEVMODE.html>

##	打印

###	无 *GDI* 打印

-	`win32print.StartDocPrinter(hPrinter, level, tuple)`：通知打印 *spool* 将有文件加入
	-	`hPrinter`：`wi32print.OpenPrinter` 返回的打印机句柄
	-	`level`：文档信息结构，此处仅支持 `1` （元组）
	-	`tuple`：按 `level` 指定格式设置的文档信息
		-	`docName`：文档名
		-	`outputFile`：输出文件名
		-	`dataType`：文档数据类型
-	`win32print.EndDocPrinter(hPrinter)`：结束打印机的打印任务
	-	在 `win32print.WriterPrinter` 调用后使用
-	`win32printer.AbortPrinter(hPrinter)`：删除打印机的 *spool* 文件
-	`win32print.StartPagePrinter(hPrinter)`：通知打印 *spool* 一页将被打印
-	`win32print.EndPagePrinter(hPrinter)`：结束打印 *job* 中的一页
-	`win32print.WritePrinter(hPrinter, buf)`：将 `buf` 复制到打印机
	-	适合复制 `raw Postscripts`、`HPGL` 文件

###	*GDI* 打印

-	`win32print.StartDoc(hdc, docinfo)`：在打印机设备上下文上开始 *spooling* 打印任务
	-	`hdc`：`win32gui.CreateDC` 返回的设备上下文句柄
	-	`docinfo`：指定打印任务参数，四元组
		-	`DocName`：文档名
		-	`Output`：输出文件名，仅在输出至文件时需要，正常打印时可设置为 `None`
		-	`DataType`：数据类型，如：`RAW`、`EMF`、TEXT`，`None` 使用默认
		-	`Type`：操作模式，可选值 `DI_APPBANDING`、`DI_ROPS_READ_DESTINATION`、`0`
-	`win32print.EndDoc(hdc, docinfo)`：在打印机设备上下文上结束 *spooling* 打印任务
-	`win32print.AbortDoc(hdc)`：取消打印 *job*
-	`win32print.StartPage(hdc)`：在打印机设备上下文上开始一页
-	`win32print.EndPage(hdc)`：在打印机设备上下文上结束一页

##	打印任务

-	`win32print.EnumJobs(hPrinter, FirstJob, NoJobs, Level)`：枚举打印机上的打印 *job*
-	`win32print.GetJob(hPrinter, JobID, Level)`：获取打印 *job* 的信息
-	`win32print.SetJob(hPrinter, JobID, Level, JobInfo, Command)`：暂停、取消、恢复、设置优先级
	-	`Command`：指定设置的内容，可以通过查询 `win32print.JOB_CONTROL_<XXXX>` 查看枚举值

##	打印示例

###	图片打印

-	直接使用 `win32gui.CreateDC`

	```python
	# 创建打印机上下文
	hdc = win32gui.CreateDC("WINSPOOL", printer_name, None)
	# 打开图像文件，并创建位图
	bmp = Image.Open(filename)
	dib = ImageWin.Dib(bmp)
	# 在上下文上打开文件、文件页
	win32print.StartDoc(hdc, 1, ("", None, None, 0))
	win32print.StartPage(hdc)
	# 绘制位图
	dib.draw(hdc, (x1, y1, x2, y2))
	win32print.EndPage(hdc)
	win32print.EndDoc(hdc)
	win32gui.DeleteDC(hdc)
	```

-	使用 `win32ui.CreateDC`

	```python
	# 创建未初始化设备上下文
	hdc = win32ui.CreateDC()
	# 初始化上下文
	hdc.CreatePrinterDC(printer_name)
	hdc.StartDoc(filename)
	hdc.StartPage()
	bmp = Image.Open(filename)
	dib = ImageWin.Dib(bmp)
	# 获取句柄，并绘制
	dib.draw(hdc.GetHandleOutput(), (x1, y1, x2, y2))
	hdc.EndPage()
	hdc.EndDoc()
	hdc.DeleteDC()
	```

-	结合 `win32gui.CreateDC`、`win32ui.CreateDC`

	```python
	# 可以设置 `PDEVMODE`
	g_hdc = win32gui.CreateDC("WINSPOOL", printer_name, dev_mode)
	hdc = win32gui.CreateDCFromHandle(g_hdc)
	# 之后同上
	```

> - 可通过 `GetDeviceCaps` 方法、函数获取设备上下文属性，用于确定、设置打印位置、方向等
> - <https://stackoverflow.com/questions/54522120/python3-print-landscape-image-file-with-specified-printer>
> - <https://www.cnblogs.com/onsunsl/p/python_call_win32print_print_unicode.html>

