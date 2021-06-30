---
title: Pywin32 应用模块
categories:
  - Python
  -	Pywin32
tags:
  - Python
  - Python32
  - Automatic
date: 2021-05-31 11:12:29
updated: 2021-05-31 11:12:29
toc: true
mathjax: true
description: 
---

#	`win32com`

##	`win32com.client`

###	`win32com.client.Dispatch`

####	Word.Application

-	创建、打开文档

	```python
	# 应用对象：word应用
	app = win32com.client.Dispatch("Word.Application")
	# 文档对象
	doc = app.Documents.add()
	doc = app.Documents.Open("/path/to/file")
	# 窗口对象：文档可以在多个窗口中展示（`View`栏中可找到）（不是打开两次文档）
	window = app.windows(winNo)
	window = doc.windows(winNo)
	# 视图对象：窗口的视图属性（全部显示，域底纹，表格虚框）
	view = window.view
	```

-	*Selection* 对象：选区，文档中高亮的选择区域、插入点
	-	每个窗口有自身的 *Selection*
	-	窗口和相应的 *Selection* 同时只有一个被激活

	```python
	# 当前激活窗口的选区
	s = app.Selection()
	# 1窗口的选区
	s1 = app.Windows(1).Selection
	# 替换选区内容，选区变为输入文本整体
	s.Text = "Hello, world!"
	# 输入内容，选区变为文本后插入点
	s.TypeText("hello, world!")
	# 拷贝选区
	s.Copy()
	# 粘贴内容至选区
	s.Paste()
	# 变换选区
	s.Start = 0
	s.End = 1
	# 删除
	s.Delete()
	# 全选
	s.WholeStory()
	# 移动选区：<wdunits>：移动单位：1-字符，
	s.MoveLeft(<wdunits>, <nums>)
	s.MoveRight()
	```

-	*Range* 对象：连续区域，由 `<start>`、`<end>` 位置定义
	-	区分文档不同部分
	-	*Range* 独立于 *Selection*
	-	可以定义多个 *Range*
	-	属性、方法类似 *Selection*

	```python
	r = doc.Range(<start>, <end>)
	r = s.Range()
	```

-	*Font* 对象：字体属性（名称、字号、颜色等）

	```python
	font = s.Font
	font = r.Font
	font.name = "仿宋"
	font.size = 16
	```

-	*ParagraphFormat* 对象：段落格式（对齐、缩进、行距、边框底纹等）

	```python
	pf = s.ParagraphFormat
	pf = r.ParagraphFormat
	# 对齐方式：0-左对齐，1-居中对齐，2-右对齐
	pf.Alignment = 0
	# 行间距：0-单倍，1-1.5倍，2-双倍
	pf.LineSpacingRule = 0
	# 左缩进：单位：磅
	pf.LeftIndent = 21
	```

-	*PageSetup* 对象：页面设置（左边距、右边距、纸张大小）

	```python
	ps = doc.PageSetup
	ps = s.PageSetup
	ps = r.PageSetup
	# 上边距：单位：磅（1cm = 28.35磅）
	cm2pound = 28.35
	ps.TopMargin = 79
	# 页面大小：6-A3，7-A4
	ps.PageSize = 7
	# 布局模式
	ps.LayoutMode = 1
	# 行字符数
	ps.CharsLine = 28
	# 行数：自动设计行间距
	ps.LinesPage = 22
	```

-	样式集：文档中内置、用户定义的样式

	```python
	styles = doc.Styles
	# 获取样式：-1-正文，-2-标题1，-3-标题2，-4-标题3，-32-页眉，-63-标题
	normal = styles(-1)
	normal.Font.Name
	normal.Font.Size = 16
	```

> - 参考资料
> > -	宏录制：查看大致方法
> > -	*Word -> 宏编辑器 -> 对象浏览器*：查询各组件方法、属性
> > -	![*.NET* 文档]<https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.office.interop.word>：查询语法
> - <https://zhuanlan.zhihu.com/p/67543981>

